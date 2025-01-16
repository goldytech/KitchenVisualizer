from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from starlette.requests import Request

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/upload-form/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    valid_types = ["image/png", "image/jpeg"]
    max_size = 5 * 1024 * 1024  # 5 MB

    if file.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    if file.spool_max_size > max_size:
        raise HTTPException(status_code=400, detail="File too large")

    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    return {"info": f"File '{file.filename}' uploaded successfully"}