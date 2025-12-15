from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import pickle
import numpy as np

class UserData(BaseModel):
    age: int = Field(example=5, description="Edad del usuario")
    visits: int = Field(example=5, description="Numero de visitas al sitio")
    time_on_site: float = Field(example=60, description="Tiempo en el sitio (en minutos)")
    cart_value: float = Field(example=50, description="Valor del carrito de compras")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "visits": 5,
                "time_on_site": 60,
                "cart_value": 50
            }
        }

app = FastAPI()

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# API Restfull

@app.get("/health", tags=["Sistema"])
def health():
    return {"status": "ok"}

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
    X = np.array([[data.age, data.visits, data.time_on_site, data.cart_value]])
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0][1]
    pred = int(proba > 0.5)
    
    return {"prediction": pred, "probability": float(proba)}