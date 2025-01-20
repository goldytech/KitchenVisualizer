from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload-form/", response_class=HTMLResponse)
async def upload_form(request: Request):
    try:
        return templates.TemplateResponse("upload.html", {"request": request})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file uploads to the server.

    Args:
        file (UploadFile): The file to be uploaded; must be provided.

    Validations:
        - File type must be either "image/png" or "image/jpeg".
        - File size must not exceed 5 MB.

    Returns:
        JSONResponse:
            - 400: If the file fails validation (includes details of the error(s)).
            - 200: If the file is uploaded successfully (includes upload information).

    Raises:
        HTTPException (500): If there is any server-side error during file processing.
    """
    valid_types = ["image/png", "image/jpeg"]
    max_size = 5 * 1024 * 1024  # 5 MB
    errors = []

    try:
        if file.content_type not in valid_types:
            errors.append("Invalid file type. Only .png and .jpeg files are allowed.")

        file_size = await file.read()
        if len(file_size) > max_size:
            errors.append("File size exceeds 5 MB.")

        if errors:
            return JSONResponse(status_code=400, content={"errors": errors})

        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file_size)

        return {"info": f"File '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))