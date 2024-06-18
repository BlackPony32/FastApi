from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
import shutil
import requests
import logging

from side_func import extract_filename, get_file_name
# Import additional necessary modules
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI()

# Constants for upload directories
UPLOAD_DIR = "uploads"
UPLOAD_DIR_MANY = "uploads_many"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR_MANY, exist_ok=True)

# List of visualization keywords
visualization_keywords = ["visualize", "plot", "graph", "chart", "draw", "diagram", "display"]

# Pydantic models for request validation
class ChatRequest(BaseModel):
    prompt: str

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

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
@app.post("/upload_many_files/")
async def upload_many_files(files: List[UploadFile] = File(...)):
    cleanup_uploads_folder(UPLOAD_DIR_MANY)
    last_uploaded_file_paths = []

    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".csv", ".xlsx"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        file_path = os.path.join(UPLOAD_DIR_MANY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file_ext == ".xlsx":
            file_path = convert_excel_to_csv(file_path)

        last_uploaded_file_paths.append(file_path)

    return JSONResponse(content={
        "message": "Files uploaded successfully",
        "file_paths": last_uploaded_file_paths,
        "streamlit_url": "http://51.20.119.227:8501/"
    })

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    cleanup_uploads_folder(UPLOAD_DIR)
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in [".csv", ".xlsx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_path = os.path.join(UPLOAD_DIR, f"temp_file{file_ext}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_ext == ".xlsx":
        file_path = convert_excel_to_csv(file_path)

    return JSONResponse(content={"message": "File uploaded successfully", "file_path": file_path})

@app.post("/link_file_and_name/")
async def link_file_and_name(request: DownloadRequest):
    cleanup_uploads_folder(UPLOAD_DIR)
    url = request.url
    filename = request.filename

    report_type_filenames = {
        'CUSTOMER_DETAILS': 'customer_details.xlsx',
        'TOP_CUSTOMERS': 'top_customers.xlsx',
        'ORDER_SALES_SUMMARY': 'order_sales_summary.xlsx',
        'THIRD_PARTY_SALES_SUMMARY': 'third_party_sales_summary.xlsx',
        'CURRENT_INVENTORY': 'current_inventory.xlsx',
        'LOW_STOCK_INVENTORY': 'low_stock_inventory.xlsx',
        'BEST_SELLERS': 'best_sellers.xlsx',
        'SKU_NOT_ORDERED': 'sku_not_ordered.xlsx',
        'REP_DETAILS': 'rep_details.xlsx',
        'REPS_SUMMARY': 'reps_summary.xlsx',
    }

    friendly_filename = report_type_filenames.get(filename, 'unknown.xlsx')

    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        if response.headers.get('Content-Type') != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            raise HTTPException(status_code=400, detail="Unsupported file type")

        excel_file_path = os.path.join(UPLOAD_DIR, friendly_filename)
        with open(excel_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        csv_file_path = convert_excel_to_csv(excel_file_path)

        return {"message": "File downloaded and converted successfully", "streamlit_url": "http://51.20.119.227:8501/"}
    except requests.RequestException as e:
        logging.error(f"RequestException: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/download")
async def download_file(request: DownloadRequest):
    cleanup_uploads_folder(UPLOAD_DIR)
    url = request.url
    file_name = extract_filename(url)

    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        if response.headers.get('Content-Type') != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            raise HTTPException(status_code=400, detail="Unsupported file type")

        excel_file_path = os.path.join(UPLOAD_DIR, f"{file_name}.xlsx")
        with open(excel_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        csv_file_path = convert_excel_to_csv(excel_file_path)

        return {"message": "File downloaded and converted successfully", "streamlit_url": "http://51.20.119.227:8501/"}
    except requests.RequestException as e:
        logging.error(f"RequestException: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.post("/chat_with_file/")
async def chat_with_file_endpoint(chat_request: ChatRequest):
    file_name = get_file_name()
    file_path = os.path.join(UPLOAD_DIR, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="No file has been uploaded or downloaded yet")

    try:
        response = chat_with_agent(chat_request.prompt, file_path)
        return {"response": response}
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def chat_with_agent(input_string, file_path):
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor
    from langchain_experimental.agents.agent_toolkits import create_csv_agent
    from langchain.agents.agent_types import AgentType
    from langchain.tools import tool
    try:
        df = pd.read_csv(file_path)

        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )

        if any(keyword in input_string.lower() for keyword in visualization_keywords):
            data_for_visualization = df.to_dict(orient='records')
            return data_for_visualization
        else:
            result = agent.invoke(input_string)
            return result
    except pd.errors.ParserError as e:
        raise ValueError(f"Parsing error occurred: {str(e)}")
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    cleanup_uploads_folder(UPLOAD_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
