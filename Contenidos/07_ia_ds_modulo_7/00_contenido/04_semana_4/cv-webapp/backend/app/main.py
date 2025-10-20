from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .model import VisionModel
from .schemas import PredictResponse, Base64ImageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_webapp.backend")

app = FastAPI(title="CV WebApp API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: VisionModel | None = None

@app.on_event("startup")
def startup_event():
    global model
    model = VisionModel(device="cpu")
    logger.info("Model loaded on startup")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo no es imagen")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande")
    result = model.predict_bytes(contents)
    return JSONResponse(content=result)

@app.post("/predict/base64", response_model=PredictResponse)
async def predict_base64(payload: Base64ImageRequest):
    try:
        result = model.predict_base64(payload.image_base64)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

@app.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo no es imagen")
    contents = await file.read()
    detections = model.detect_bytes(contents, threshold=0.5)
    return JSONResponse(content={"detections": detections})
