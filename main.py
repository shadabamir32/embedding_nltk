from typing import Union, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from enums.embeddingmodels import EmbeddingModels;
from backend.handler import train_model, preprocess_json, cosin_lookup
from utils.json_handler import convert_jsonl_to_json
import hashlib
import os
import shutil

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"

@app.post(path='/api/v1/upload', tags=['Upload File'])
async def upload_data_File(file: UploadFile, background_task :BackgroundTasks):
    file_hash = uploadFile(file=file)
    background_task.add_task(preprocess_json, file_hash)

    return {
        'status': 'Success',
        'message': 'File will be processed within few minutes.',
        'data': {
            'file_hash': file_hash
        },
        'code': 200
    }

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
async def train(model: EmbeddingModels, file_hash: str):
    if not os.path.exists(f"{UPLOAD_DIR}/{file_hash}"):
        raise HTTPException(status_code=404, detail="File not found")
    train_model(file_hash, model)

    return {"status": "Successs", "Message": "Model training has been completed."}

@app.get(path='/api/v1/lookup/{model}', tags=["Train Model"])
async def lookup(model: EmbeddingModels, file_hash: str, query: str):
    data = cosin_lookup(query, file_hash, model)
    return {"status": "Successs", "Data": data}

# Will work on last
# @app.get(path="/api/v1/generate/{model}", tags=["Generate"], description="Generate embedding using selected model.")
# async def generate_embeddings(model: EmbeddingModels, text: str = None):
#     return {'selectedModel': model.title(), 'payload': text}