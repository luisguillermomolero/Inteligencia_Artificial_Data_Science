from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging

from model import ImageClassifier
from schemas import ClassifyResponse, Prediction

logger = logging.getLogger("cv_app")

app = FastAPI()

classifier: ImageClassifier | None = None

def get_classifier():
    
    global classifier
    
    if classifier is None:
        logger.info("Cargando modelo por primera vez...")
        classifier = ImageClassifier(device="cpu")
        logger.info("Modelo cargado con éxito...")
    
    return classifier

@app.get("/")
def root():
    return {
        "message": "API de Clasificación de Imágenes con IA",  # Mensaje principal
        "version": "1.0.0",                                    # Versión informativa
        "description": "Clasifica imágenes usando MobileNet pre-entrenado en ImageNet",
        # Información de los endpoints disponibles
        "endpoints": {
            "health": "/health - Estado del servidor",
            "classify": "/predict/classify - Clasificar una imagen",
            "test_labels": "/test-labels - Ver etiquetas del modelo",
            "docs": "/docs - Documentación interactiva generada por FastAPI"
        },
        # Ejemplo de uso para usuarios nuevos
        "usage": {
            "method": "POST",
            "endpoint": "/predict/classify",
            "content_type": "multipart/form-data",
            "parameter": "file (imagen)"
        },
        # Ejemplo práctico de cómo consumir el servicio
        "example": "Sube una imagen a /predict/classify para obtener las 3 predicciones más probables"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-labels")
def test_labels():
    classifier = get_classifier()
    sample_labels =classifier.labels[:10]
    return {
        "labels_sample": sample_labels,
        "total_labels": len(classifier.labels)
    }

@app.post("/predict/classify", response_model=ClassifyResponse)
async def predict_classify(file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen")
    
    image_byte = await file.read()
    
    if len(image_byte) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    
    classifier = get_classifier()
    
    preds = classifier.predict(image_byte, topk=3)
    
    response = ClassifyResponse(predictions=[Prediction(label=p[0], score=p[1]) for p in preds])
    
    return response