from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import os
import pandas as pd
import json
from typing import List, Optional
from tempfile import NamedTemporaryFile
import shutil
import requests
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from app import main_viz
app = FastAPI()
load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variable to store the path of the last uploaded/downloaded file
last_uploaded_file_path = None

# List of synonyms for the word "visualize"
visualization_keywords = ["visualize", "plot", "graph", "chart", "draw", "diagram", "display"]

class ChatRequest(BaseModel):
    prompt: str

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    global last_uploaded_file_path
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".csv", ".xlsx"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        file_path = os.path.join(UPLOAD_DIR, f"last_uploaded{file_ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file_ext == ".xlsx":
            csv_file_path = convert_excel_to_csv(file_path)
            last_uploaded_file_path = csv_file_path
        else:
            last_uploaded_file_path = file_path
            
        return JSONResponse(content={"message": "File uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download")
def download_file(request: DownloadRequest):
    global last_uploaded_file_path
    url = request.url

    try:
        # Log the URL for debugging
        logging.info(f"Downloading file from URL: {url}")

        # Validate the URL
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Determine the MIME type
        mime_type = response.headers.get('Content-Type')
        if mime_type not in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/csv']:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Determine the file extension based on MIME type
        if mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            file_ext = '.xlsx'
        elif mime_type == 'text/csv':
            file_ext = '.csv'
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Save the file
        file_path = os.path.join(UPLOAD_DIR, f"last_uploaded{file_ext}")
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Convert Excel to CSV if necessary
        if file_ext == '.xlsx':
            csv_file_path = convert_excel_to_csv(file_path)
            last_uploaded_file_path = csv_file_path
        else:
            last_uploaded_file_path = file_path

        return {"message": "File downloaded successfully", "filename": f"last_uploaded{file_ext}"}
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/chat_with_file/")
async def chat_with_file_endpoint(chat_request: ChatRequest):
    global last_uploaded_file_path
    try:
        if last_uploaded_file_path is None or not os.path.exists(last_uploaded_file_path):
            raise HTTPException(status_code=400, detail="No file has been uploaded or downloaded yet")
            
        result = chat_with_agent(chat_request.prompt, last_uploaded_file_path)
        
        return {"response": result}
    except json.JSONDecodeError as e:
        return {"error": f"JSONDecodeError: {str(e)}"}
    except ValidationError as e:
        return {"error": f"ValidationError: {str(e)}"}
    except ValueError as e:
        return {"error": f"ValueError: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def convert_excel_to_csv(excel_file_path):
    try:
        df = pd.read_excel(excel_file_path)
        csv_file_path = os.path.splitext(excel_file_path)[0] + ".csv"
        df.to_csv(csv_file_path, index=False)
        return csv_file_path
    except Exception as e:
        raise ValueError(f"Error converting Excel to CSV: {str(e)}")

def chat_with_agent(input_string, file_path):
    try:
        # Assuming file_path is always CSV after conversion
        df = pd.read_csv(file_path)

        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-4o"),
            file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        
        # Check if any of the visualization keywords are in the input string
        if any(keyword in input_string.lower() for keyword in visualization_keywords):
            # Extract relevant data for visualization
            data_for_visualization = df.to_dict(orient='records')
            return data_for_visualization
        else:
            result = agent.invoke(input_string)
            return result
    except ImportError as e:
        raise ValueError("Missing optional dependency 'tabulate'. Use pip or conda to install tabulate.")
    except pd.errors.ParserError as e:
        raise ValueError("Parsing error occurred: " + str(e))
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

def cleanup_uploads_folder():
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up uploads folder: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    cleanup_uploads_folder()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
