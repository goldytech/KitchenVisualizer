from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
from yolo_inference import save_inference_result_image_no_boxes

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

        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file_size)

        # Run YOLO model inference
        output_image_path = f"../uploads/detected_{file.filename}"
        model_path = "best.pt"
        save_inference_result_image_no_boxes(file_location, model_path, output_image_path)

        return {"info": f"File '{file.filename}' uploaded successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"errors": [str(e)]})