from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from utils import load_module, bytes_to_tensor, post_result
import io

app = FastAPI()

model = {}

@app.on_event("startup")
async def startup():
    model['model'] = load_module()

@app.get("/api")
async def main():
    return "hello world"

@app.post("/api/uploadfile")
async def create_upload_file(file: UploadFile):
    data = await file.read()
    fp = io.BytesIO(data)
    emotion_mapping = {
        0: 'Neutral',
        1: 'Angry',
        2: 'Happy',
        3: 'Sad',
        4: 'Surprise'
    }
    result = model['model'](bytes_to_tensor(fp))
    result = post_result(result)
    del fp
    return {"filename": file.filename, "result": emotion_mapping[result]}