from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
import shutil
import requests
import logging

from side_func import extract_filename, get_file_name
# Import additional necessary modules
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from fastapi import FastAPI, Query
# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI()
# Allow all origins to access the API (change it to specific origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Constants for upload directories
UPLOAD_DIR = "uploads"
UPLOAD_DIR_MANY = "uploads_many"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR_MANY, exist_ok=True)


# Pydantic models for request validation
class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None
    
url = None
file_name = None

# Utility functions
def cleanup_uploads_folder(upload_dir: str):
    try:
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        logging.error(f"Error cleaning up uploads folder: {str(e)}")

def convert_excel_to_csv(excel_file_path):
    try:
        df = pd.read_excel(excel_file_path)
        csv_file_path = os.path.splitext(excel_file_path)[0] + ".csv"
        df.to_csv(csv_file_path, index=False)
        os.remove(excel_file_path)
        return csv_file_path
    except Exception as e:
        raise ValueError(f"Error converting Excel to CSV: {str(e)}")

# Endpoint implementations
@app.post("/link_for_streamlit/")
async def link_for_streamlit():
    streamlit_url = "https://streamlit-2y3qx63wua-uc.a.run.app/"
    return {"message": "File downloaded and converted successfully", "streamlit_url": streamlit_url}


@app.post("/link_file_and_name/")
async def link_file_and_name(request: DownloadRequest):
    global url
    global file_name
    url = request.url
    file_name = request.filename
    return JSONResponse(content={"url": url, "file_name": file_name})

@app.get("/get_file_info/")
async def get_file_info():
    if url and file_name:
        return JSONResponse(content={"url": url, "file_name": file_name})
    else:
        return JSONResponse(content={"error": "No data available"}, status_code=404)



@app.on_event("shutdown")
def shutdown_event():
    cleanup_uploads_folder(UPLOAD_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
