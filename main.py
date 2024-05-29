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

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.tools import tool

app = FastAPI()
load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variable to store the path of the last uploaded file
last_uploaded_file_path = None

# List of synonyms for the word "visualize"
visualization_keywords = ["visualize", "plot", "graph", "chart", "draw", "diagram", "display"]

class ChatRequest(BaseModel):
    prompt: str

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    global last_uploaded_file_path
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        file_path = os.path.join(UPLOAD_DIR, "last_uploaded.csv")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        last_uploaded_file_path = file_path
        return JSONResponse(content={"message": "File uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat_with_csv/")
async def chat_with_csv_endpoint(chat_request: ChatRequest):
    global last_uploaded_file_path
    try:
        if last_uploaded_file_path is None or not os.path.exists(last_uploaded_file_path):
            raise HTTPException(status_code=400, detail="No CSV file has been uploaded yet")
            
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


def chat_with_agent(input_string, file_path):
    try:
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
