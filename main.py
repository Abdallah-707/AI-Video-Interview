from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI(title="AI Recruitment CV API")

# 1. Health check route so you can test the URL in your browser easily
@app.get("/")
def health_check():
    return {"status": "Server is alive and waiting for videos!"}

@app.post("/api/analyze-interview")
async def process_interview_video(file: UploadFile = File(...)):
    # 2. ULTIMATE LAZY LOAD: We import the heavy CV stuff ONLY when a video arrives
    from interview_cv_v3 import InterviewAnalyzer 
    
    temp_video_path = f"temp_{file.filename}"
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        analyzer = InterviewAnalyzer(debug=False)
        report = analyzer.run(video_path=temp_video_path, candidate_name="Android_Candidate")
        
        return {"status": "success", "data": report}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
