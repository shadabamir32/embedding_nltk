from typing import Union, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from enums.embeddingmodels import EmbeddingModels;
from backend.handler import train_model, preprocess_json, cosin_lookup
from utils.json_handler import convert_jsonl_to_json
import redis
import hashlib
import os
import shutil
from fastapi.responses import JSONResponse

app = FastAPI(title="Document Embedding API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# r = redis.Redis(host='localhost', port=6379, decode_responses=True)
UPLOAD_DIR = "uploads"

@app.post(path='/api/v1/upload', tags=['Upload File'])
async def upload_data_File(file: UploadFile, background_task :BackgroundTasks):
    try:
        file_hash = uploadFile(file=file)
        if not os.path.exists(f"processed_files/{file_hash}/output.json"):
            background_task.add_task(preprocess_json, file_hash)
        return JSONResponse(
            status_code=200,
            content={
            'status': 'Success',
            'message': 'File uploaded successfully. Processing started.',
            'data': {
                'file_hash': file_hash
            },
            'code': 200
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'Error',
                'message': f'An error occurred: {str(e)}',
                'data': None,
                'code': 500
            }
        )

# Upload file in the upload directory.
def uploadFile(file: UploadFile) -> str:
    file_hash = get_file_hash(file.file)
    ext = os.path.splitext(file.filename)[-1]
    saved_filename = f"{file_hash}{ext}"
    uploadPath = f"{UPLOAD_DIR}/{file_hash}"
    os.makedirs(uploadPath, exist_ok=True)
    if not os.path.exists(f"{uploadPath}/{saved_filename}"):
        saved_path = os.path.join(uploadPath, saved_filename)
        # Save file
        with open(saved_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

    return file_hash

# Generate unique has for a file.
def get_file_hash(file_obj: File) -> str:
    hasher = hashlib.sha256()
    file_obj.seek(0)
    while chunk := file_obj.read(8192):
        hasher.update(chunk)
    file_obj.seek(0)

    return hasher.hexdigest()

@app.post(path='/api/v1/train/{model}', tags=["Train Model"])
async def train(model: EmbeddingModels, file_hash: str = Form(...)):
    try:
        train_model(file_hash, model)
        return JSONResponse(
            status_code=200,
            content={
            'status': 'Success',
            'message': 'Model training has been completed.',
            'data': None,
            'code': 200
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'Error',
                'message': f'An error occurred: {str(e)}',
                'data': None,
                'code': 500
            }
        )

@app.get(path='/api/v1/lookup/{model}', tags=["Train Model"])
async def lookup(model: EmbeddingModels, file_hash: str, query: str):
    try:
        data = cosin_lookup(query, file_hash, model)
        return JSONResponse(
            status_code=200,
            content={
            'status': 'Success',
            'message': 'Look up found.',
            'data': data,
            'code': 200
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'Error',
                'message': f'An error occurred: {str(e)}',
                'data': None,
                'code': 500
            }
        )