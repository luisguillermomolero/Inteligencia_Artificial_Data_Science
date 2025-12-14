# =============================
# API.PY - API REST PARA PREDICCIONES DEL MODELO
# =============================
# Este archivo crea una API REST usando FastAPI que permite hacer predicciones
# en tiempo real sobre si un usuario realizará una compra.
#
# CÓMO PROBAR LA API CON SWAGGER:
# 1. Ejecuta el servidor: uvicorn api:app --reload
# 2. Abre tu navegador en: http://localhost:8000/docs
# 3. En Swagger UI podrás ver todos los endpoints y probarlos directamente
# 4. Haz clic en "Try it out" en cualquier endpoint para probarlo
#
# EJEMPLO DE DATOS PARA PROBAR:
# {
#   "age": 35,
#   "visits": 5,
#   "time_on_site": 60,
#   "cart_value": 50
# }

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import pickle
import numpy as np

# =============================
# MODELO DE DATOS DE ENTRADA
# =============================
# Define la estructura de datos que espera la API para hacer predicciones
class UserData(BaseModel):
    age: float = Field(example=35, description="Edad del usuario")  # Edad del usuario
    visits: float = Field(example=5, description="Número de visitas al sitio")  # Número de visitas al sitio
    time_on_site: float = Field(example=60, description="Tiempo en el sitio (en minutos)")  # Tiempo en el sitio (en minutos)
    cart_value: float = Field(example=50, description="Valor del carrito de compras")  # Valor del carrito de compras
    
    class Config:
        # Ejemplo completo para la documentación de Swagger
        json_schema_extra = {
            "example": {
                "age": 35,
                "visits": 5,
                "time_on_site": 60,
                "cart_value": 50
            }
        }

# =============================
# INICIALIZACIÓN DE LA APLICACIÓN
# =============================
# Creamos la instancia de FastAPI con título y descripción
# Estos aparecerán en la documentación de Swagger
app = FastAPI(
    title="API de Predicción de Compras",
    description="API para predecir si un usuario realizará una compra basándose en sus características",
    version="1.0.0"
)

# =============================
# CARGA DEL MODELO Y PREPROCESADOR
# =============================
# Cargamos el modelo entrenado y el scaler al iniciar la aplicación
# Estos se cargan una sola vez cuando se inicia el servidor
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =============================
# ENDPOINT: HEALTH CHECK
# =============================
# Endpoint simple para verificar que la API está funcionando
@app.get("/health", tags=["Sistema"])
def health():
    """
    Verifica el estado de la API.
    
    Returns:
        dict: Estado de la API (siempre retorna {"status": "ok"})
    """
    return {"status": "ok"}

# =============================
# ENDPOINT: PREDICCIÓN
# =============================
# Endpoint principal que recibe datos del usuario y retorna una predicción
@app.post("/predict", tags=["Predicciones"])
def predict(
    data: UserData = Body(
        example={
            "age": 35,
            "visits": 5,
            "time_on_site": 60,
            "cart_value": 50
        }
    )
):
    """
    Predice si un usuario realizará una compra basándose en sus características.
    
    Args:
        data (UserData): Datos del usuario (edad, visitas, tiempo en sitio, valor del carrito)
    
    Returns:
        dict: Diccionario con:
            - prediction: 1 si se predice compra, 0 si no
            - probability: Probabilidad de compra (0.0 a 1.0)
    
    Ejemplo de uso en Swagger:
        1. Haz clic en "Try it out"
        2. Modifica los valores en el JSON de ejemplo:
           {
             "age": 35,
             "visits": 5,
             "time_on_site": 60,
             "cart_value": 50
           }
        3. Haz clic en "Execute"
        4. Verás la respuesta con la predicción y probabilidad
    """
    # Convertimos los datos del usuario a un array numpy para el modelo
    X = np.array([[data.age, data.visits, data.time_on_site, data.cart_value]])
    
    # Normalizamos los datos usando el mismo scaler del entrenamiento
    X_scaled = scaler.transform(X)
    
    # Obtenemos la probabilidad de compra (clase 1)
    proba = model.predict_proba(X_scaled)[0][1]
    
    # Convertimos la probabilidad en una predicción binaria (1 si > 0.5, 0 si no)
    pred = int(proba > 0.5)
    
    # Retornamos la predicción y la probabilidad
    return {"prediction": pred, "probability": float(proba)}
