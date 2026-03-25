import cv2
import mediapipe as mp
import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict
from deepface import DeepFace
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#  CONFIG  — tune these to adjust sensitivity
# ══════════════════════════════════════════════════════
INTERVIEW_DURATION   = 0      # 0 = unlimited
EMOTION_EVERY_N_SEC  = 3
YOLO_EVERY_N_FRAMES  = 10     # run YOLO every 10 frames
YOLO_CONF            = 0.40   # lower = more sensitive (phone detection needs this)
YOLO_MODEL           = "yolov8s.pt"  # s is better than n for small objects like phones
GAZE_THRESHOLD       = 0.38
MULTI_FACE_LIMIT     = 1
CONFIRM_RUNS_NEEDED  = 2
PENALTY_DRIP_RATE    = 0.15
PRESENCE_ABSENT_DRIP = 0.08
GAZE_H_THRESHOLD     = 0.18   
GAZE_V_THRESHOLD     = 0.22   
GAZE_H_PENALTY_DRIP  = 0.06   
GAZE_V_PENALTY_DRIP  = 0.02   
EMOTION_SMOOTH_N     = 5

IGNORED_CLASSES = {
    "person", "chair", "couch", "bed", "dining table",
    "toilet", "sink", "refrigerator", "microwave", "oven",
    "toaster", "hair drier", "toothbrush", "wall", "ceiling",
}

DISTRACTION_DB = {
    "cell phone"   : {"severity": "CRITICAL", "penalty": 15, "emoji": "📱"},
    "laptop"       : {"severity": "CRITICAL", "penalty": 15, "emoji": "💻"},
    "tablet"       : {"severity": "CRITICAL", "penalty": 14, "emoji": "📟"},
    "tv"           : {"severity": "CRITICAL", "penalty": 12, "emoji": "📺"},
    "remote"       : {"severity": "CRITICAL", "penalty": 10, "emoji": "📡"},
    "keyboard"     : {"severity": "CRITICAL", "penalty": 8,  "emoji": "⌨️"},
    "mouse"        : {"severity": "CRITICAL", "penalty": 8,  "emoji": "🖱️"},
    "book"         : {"severity": "HIGH",     "penalty": 10, "emoji": "📚"},
    "notebook"     : {"severity": "HIGH",     "penalty": 10, "emoji": "📓"},
    "magazine"     : {"severity": "HIGH",     "penalty": 8,  "emoji": "📰"},
    "clock"        : {"severity": "MEDIUM",   "penalty": 3,  "emoji": "🕐"},
    "cup"          : {"severity": "LOW",      "penalty": 1,  "emoji": "☕"},
    "bottle"       : {"severity": "LOW",      "penalty": 1,  "emoji": "🍶"},
    "wine glass"   : {"severity": "LOW",      "penalty": 2,  "emoji": "🍷"},
    "bowl"         : {"severity": "LOW",      "penalty": 1,  "emoji": "🥣"},
    "cat"          : {"severity": "LOW",      "penalty": 2,  "emoji": "🐱"},
    "dog"          : {"severity": "LOW",      "penalty": 2,  "emoji": "🐶"},
    "backpack"     : {"severity": "LOW",      "penalty": 2,  "emoji": "🎒"},
    "scissors"     : {"severity": "LOW",      "penalty": 2,  "emoji": "✂️"},
}

LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]
LEFT_EYE    = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_L  = 33
RIGHT_EYE_R = 263

class ObjectTracker:
    def __init__(self):
        self.active      = {}
        self.durations   = defaultdict(float)
        self.appearances = defaultdict(int)
        self.timeline    = []

    def update(self, confirmed_names, now):
        s = set(confirmed_names)
        for n in s:
            if n not in self.active:
                self.active[n] = now
                self.appearances[n] += 1
                self.timeline.append((now, "APPEARED", n))
        for n in list(self.active):
            if n not in s:
                self.durations[n] += now - self.active[n]
                self.timeline.append((now, "LEFT", n))
                del self.active[n]

    def finalize(self, now):
        for n, t in self.active.items():
            self.durations[n] += now - t
        self.active = {}

class InterviewAnalyzer:
    def __init__(self, debug=False):
        self.debug = debug
        print("\n[INFO] Loading models...")

        # --- LAZY IMPORT YOLO HERE ---
        from ultralytics import YOLO

        mp_fm = mp.solutions.face_mesh
        mp_fd = mp.solutions.face_detection
        self.mp_draw        = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        self.mp_fm_ref      = mp_fm

        self.face_mesh   = mp_fm.FaceMesh(
            max_num_faces=4, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_detect = mp_fd.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        self.yolo = YOLO(YOLO_MODEL)

        self.stats = {
            "frames_total"        : 0,
            "frames_face_present" : 0,
            "frames_multi_face"   : 0,
            "frames_no_face"      : 0,
            "frames_eye_contact"  : 0,
            "frames_looking_away" : 0,
            "head_pose_counts"    : {"Forward":0,"Left":0,"Right":0,"Up":0,"Down":0},
            "emotions"            : [],
        }
        self._presence_penalty   = 0.0   
        self._gaze_penalty       = 0.0   
        self._emotion_history    = []    
        self.obj_tracker     = ObjectTracker()
        self._run_counter    = defaultdict(int)
        self._confirmed_now  = set()
        self._dist_penalty   = 0.0

        self.session_start    = time.time()
        self.last_emotion_t   = 0
        self.frame_count      = 0

    def get_gaze(self, lm, W, H):
        def ic(idx):
            return np.array([[lm[i].x*W, lm[i].y*H] for i in idx]).mean(0)
        def corners(idx):
            pts = np.array([[lm[i].x*W, lm[i].y*H] for i in idx])
            return pts[pts[:,0].argmin()], pts[pts[:,0].argmax()]
        try:
            li = ic(LEFT_IRIS);  ri = ic(RIGHT_IRIS)
            ll,lr = corners(LEFT_EYE); rl,rr = corners(RIGHT_EYE)
            gx = ((li[0]-ll[0])/(lr[0]-ll[0]+1e-6)-0.5 +
                  (ri[0]-rl[0])/(rr[0]-rl[0]+1e-6)-0.5) / 2
            lvy = (li[1]-lm[386].y*H)/(lm[374].y*H-lm[386].y*H+1e-6)-0.5
            rvy = (ri[1]-lm[159].y*H)/(lm[145].y*H-lm[159].y*H+1e-6)-0.5
            gy  = (lvy+rvy)/2

            h_dev = abs(gx) > GAZE_H_THRESHOLD
            v_dev = abs(gy) > GAZE_V_THRESHOLD
            contact = not h_dev and not v_dev

            if   h_dev and gx > 0: direction = "Right"
            elif h_dev and gx < 0: direction = "Left"
            elif v_dev and gy > 0: direction = "Down"
            elif v_dev and gy < 0: direction = "Up"
            else:                  direction = "Center"

            return gx, gy, contact, h_dev, v_dev, direction
        except Exception:
            return 0.0, 0.0, True, False, False, "Center"

    def get_head_pose(self, lm, W, H):
        try:
            nose    = np.array([lm[NOSE_TIP].x*W,    lm[NOSE_TIP].y*H])
            le      = np.array([lm[LEFT_EYE_L].x*W,  lm[LEFT_EYE_L].y*H])
            re      = np.array([lm[RIGHT_EYE_R].x*W, lm[RIGHT_EYE_R].y*H])
            chin    = np.array([lm[CHIN].x*W,         lm[CHIN].y*H])
            mid     = (le+re)/2
            hx = (nose[0]-mid[0]) / (abs(re[0]-le[0])+1e-6)
            vy = (nose[1]-mid[1]) / (abs(chin[1]-mid[1])+1e-6)
            if   hx >  0.15: return "Left"
            elif hx < -0.15: return "Right"
            elif vy <  0.35: return "Up"
            elif vy >  0.60: return "Down"
            else:            return "Forward"
        except Exception:
            return "Forward"

    def get_emotion(self, frame, face_bbox=None):
        # --- LAZY IMPORT DEEPFACE HERE ---
        from deepface import DeepFace

        try:
            H, W = frame.shape[:2]
            if face_bbox:
                x1,y1,x2,y2 = face_bbox
                pad = 20
                x1 = max(0, x1-pad); y1 = max(0, y1-pad)
                x2 = min(W, x2+pad); y2 = min(H, y2+pad)
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] < 80 or crop.shape[1] < 80:
                    crop = cv2.resize(crop, (160,160))
            else:
                crop = frame

            r = DeepFace.analyze(crop, actions=["emotion"],
                                 enforce_detection=False, silent=True)
            r = r[0] if isinstance(r, list) else r
            raw = r.get("dominant_emotion", "neutral")
        except Exception:
            raw = "neutral"

        self._emotion_history.append(raw)
        if len(self._emotion_history) > EMOTION_SMOOTH_N:
            self._emotion_history.pop(0)

        counts = {}
        for e in self._emotion_history:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def run_yolo(self, frame):
        seen_this_run = set()
        try:
            results = self.yolo(frame, conf=YOLO_CONF, verbose=False)[0]
            for box in results.boxes:
                raw_name = results.names[int(box.cls[0])]
                name     = raw_name.lower()
                
                if name in IGNORED_CLASSES or name not in DISTRACTION_DB:
                    continue
                seen_this_run.add(name)

        except Exception as e:
            if self.debug: print(f"[YOLO] Error: {e}")

        for name in seen_this_run:
            self._run_counter[name] += 1

        for name in list(self._run_counter.keys()):
            if name not in seen_this_run:
                self._run_counter[name] = 0
                self._confirmed_now.discard(name)

        confirmed = []
        for name, count in self._run_counter.items():
            if count >= CONFIRM_RUNS_NEEDED:
                confirmed.append(name)
                self._confirmed_now.add(name)

        return confirmed

    def presence_score(self): return max(0.0, 30.0 - min(30.0, self._presence_penalty))
    def gaze_score(self): return max(0.0, 35.0 - min(35.0, self._gaze_penalty))
    def head_score(self):
        P = max(1, self.stats["frames_face_present"])
        F = self.stats["head_pose_counts"]["Forward"]
        return (F/P) * 20.0
    def multi_face_penalty(self):
        T = max(1, self.stats["frames_total"])
        M = self.stats["frames_multi_face"]
        return min(15.0, (M/T)*40.0)
    def face_score(self):
        return max(0.0, self.presence_score() + self.gaze_score() + self.head_score() - self.multi_face_penalty())
    def dist_score(self): return max(0.0, 100.0 - min(100.0, self._dist_penalty))
    def unified_score(self): return round(self.face_score()*0.65 + self.dist_score()*0.35, 1)

    def generate_report(self, name):
        now = time.time()
        self.obj_tracker.finalize(now)
        T   = max(1,self.stats["frames_total"])
        P   = max(1,self.stats["frames_face_present"])
        emo = {}
        for e in self.stats["emotions"]:
            emo[e] = emo.get(e,0)+1
        dom_emo = max(emo,key=emo.get) if emo else "neutral"

        objs = {}
        for n in self.obj_tracker.appearances:
            info = DISTRACTION_DB.get(n,{})
            objs[n] = {
                "severity"       : info.get("severity","LOW"),
                "appearances"    : self.obj_tracker.appearances[n],
                "visible_seconds": round(self.obj_tracker.durations[n],1),
                "penalty"        : info.get("penalty",2),
            }

        return {
            "candidate"        : name,
            "timestamp"        : datetime.now().isoformat(),
            "duration_seconds" : int(now-self.session_start),
            "scores": {
                "unified"      : self.unified_score(),
                "face"         : round(self.face_score(),1),
                "distraction"  : round(self.dist_score(),1),
            },
            "face_metrics": {
                "presence_pct"      : round(self.stats["frames_face_present"]/T*100,1),
                "eye_contact_pct"   : round(self.stats["frames_eye_contact"]/P*100,1),
                "multi_face_pct"    : round(self.stats["frames_multi_face"]/T*100,1),
                "head_pose"         : self.stats["head_pose_counts"],
                "dominant_emotion"  : dom_emo,
            },
            "distraction_metrics": {
                "total_penalty"  : round(self._dist_penalty,2),
                "objects"        : objs,
            },
            "flags": {
                "cheating_risk"          : self.stats["frames_multi_face"]>10,
                "electronic_device_found": any(n in ["cell phone","laptop","tablet","tv"] for n in objs),
            }
        }

    def run(self, video_path, candidate_name="Candidate"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return {"error": "Could not open video file"}

        self.session_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break  # End of video file

            frame = cv2.flip(frame, 1) # Optional for recorded video, kept for consistency
            H,W   = frame.shape[:2]
            self.frame_count += 1
            self.stats["frames_total"] += 1
            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            det = self.face_detect.process(rgb)
            face_count  = len(det.detections) if det.detections else 0
            face_present= face_count >= 1
            multi_face  = face_count > MULTI_FACE_LIMIT

            if face_present:
                self.stats["frames_face_present"] += 1
            else:
                self.stats["frames_no_face"] += 1
                self._presence_penalty = min(30.0, self._presence_penalty + PRESENCE_ABSENT_DRIP)
            if multi_face: self.stats["frames_multi_face"] += 1

            face_bbox = None
            mesh = self.face_mesh.process(rgb)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                cur_gaze_x, cur_gaze_y, cur_eye_contact, cur_h_dev, cur_v_dev, cur_gaze_dir = self.get_gaze(lm, W, H)
                cur_head_pose = self.get_head_pose(lm, W, H)

                if cur_eye_contact:
                    self.stats["frames_eye_contact"] += 1
                else:
                    self.stats["frames_looking_away"] += 1
                    if cur_h_dev: self._gaze_penalty = min(35.0, self._gaze_penalty + GAZE_H_PENALTY_DRIP)
                    if cur_v_dev: self._gaze_penalty = min(35.0, self._gaze_penalty + GAZE_V_PENALTY_DRIP)

                self.stats["head_pose_counts"][cur_head_pose] += 1
                xs = [lm[i].x*W for i in range(0,468)]
                ys = [lm[i].y*H for i in range(0,468)]
                face_bbox = (int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys)))

            if now-self.last_emotion_t >= EMOTION_EVERY_N_SEC and face_present:
                cur_emotion = self.get_emotion(frame, face_bbox)
                self.stats["emotions"].append(cur_emotion)
                self.last_emotion_t = now

            if self.frame_count % YOLO_EVERY_N_FRAMES == 0:
                cur_confirmed = set(self.run_yolo(frame))
                self.obj_tracker.update(cur_confirmed, now)
                for name in cur_confirmed:
                    info = DISTRACTION_DB.get(name, {})
                    drip = info.get("penalty", 2) * PENALTY_DRIP_RATE
                    self._dist_penalty = min(100.0, self._dist_penalty + drip)

        cap.release()
        return self.generate_report(candidate_name)
