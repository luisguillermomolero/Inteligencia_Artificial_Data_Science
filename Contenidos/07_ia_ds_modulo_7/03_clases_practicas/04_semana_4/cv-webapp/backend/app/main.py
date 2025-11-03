from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .model import VisionModel
from .schemas import PredictResponse
from .config import ALLOWED_ORIGINS, MAX_UPLOAD_BYTES, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_webapp.backend")

model: VisionModel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = VisionModel(device=DEVICE)
    logger.info("Model loaded on startup")
    yield
    # Cleanup (opcional)
    model = None

app = FastAPI(title="CV WebApp API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo no es imagen")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande")
    result = model.predict_bytes(contents)
    return JSONResponse(content=result)


