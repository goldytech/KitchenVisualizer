from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from app.api.endpoints import router as api_router

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.include_router(api_router)