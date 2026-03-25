from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from interview_cv_v3 import InterviewAnalyzer

app = FastAPI(title="AI Recruitment CV API")

@app.post("/api/analyze-interview")
async def process_interview_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    try:
        # 1. Save the uploaded video temporarily
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Initialize and run the CV analysis
        analyzer = InterviewAnalyzer(debug=False)
        # You can eventually pass the candidate's real name from the Android app
        report = analyzer.run(video_path=temp_video_path, candidate_name="Android_Candidate")
        
        # 3. Return the JSON report directly
        return {"status": "success", "data": report}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        
    finally:
        # 4. Cleanup: Always delete the video file from the server when done
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
