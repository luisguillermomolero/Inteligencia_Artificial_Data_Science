from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pickle
from pathlib import Path
import uvicorn

MODEL_PATH = Path("models/model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

FEATURE_COLS = [
    "age",
    "tenure",
    "monthly_fee",
    "num_products"
    "has_partner",
    "has_dependents",
]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    scaler = pickle.load(f)

class CustomerData(BaseModel):
    age: float
    tenure: float
    monthly_fee: float
    num_products: float
    has_partner: Literal[0, 1]
    has_dependents: Literal[0, 1]

app = FastAPI()

app.post("/predict")
def predict_churn(data: CustomerData):
    X = [[
        data.age,
        data.tenure,
        data.monthly_fee,
        data.num_products,
        data.has_partner, 
        data.has_dependents,
    ]]
    
    X_scaled = scaler.transform(X)
    
    proba_churn = model.predict_proba(X_scaled)[0, 1]
    
    prediction = int(proba_churn > 0.5)
    
    return{
        "churn_probability": float(proba_churn),
        "churn_prediction": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)