# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging

from model.model import ImageClassifier
from schemas.schema import ClassifyResponse, Prediction

logger = logging.getLogger("cv_app")
app = FastAPI(title="CV Model API", version="0.1")

# cargamos el modelo de forma lazy (cuando se necesite)
classifier: ImageClassifier | None = None

def get_classifier():
    global classifier
    if classifier is None:
        logger.info("Cargando modelo por primera vez...")
        classifier = ImageClassifier(device="cpu")
        logger.info("Modelo cargado exitosamente")
    return classifier


@app.get("/")
def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Clasificación de Imágenes con IA",
        "version": "1.0.0",
        "description": "Clasifica imágenes usando MobileNet pre-entrenado en ImageNet",
        "endpoints": {
            "health": "/health - Estado del servidor",
            "classify": "/predict/classify - Clasificar una imagen",
            "test_labels": "/test-labels - Ver etiquetas del modelo",
            "docs": "/docs - Documentación interactiva"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/predict/classify",
            "content_type": "multipart/form-data",
            "parameter": "file (imagen)"
        },
        "example": "Sube una imagen a /predict/classify para obtener las 3 predicciones más probables"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-labels")
def test_labels():
    """Endpoint para probar las etiquetas del modelo"""
    classifier = get_classifier()
    # Mostrar las primeras 10 etiquetas
    sample_labels = classifier.labels[:10]
    return {"labels_sample": sample_labels, "total_labels": len(classifier.labels)}

@app.post("/predict/classify", response_model=ClassifyResponse)
async def predict_classify(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo no es una imagen")
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    classifier = get_classifier()  # Carga el modelo solo cuando se necesita
    preds = classifier.predict(image_bytes, topk=3)  # Reducido de 5 a 3 predicciones
    response = ClassifyResponse(predictions=[Prediction(label=p[0], score=p[1]) for p in preds])
    return response
