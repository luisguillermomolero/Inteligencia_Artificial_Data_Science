from pydantic import BaseModel, Field
from typing import List, Tuple

class Prediction(BaseModel):
    label: str
    score: float

class ClassifyResponse(BaseModel):
    status: str = Field("ok")
    predictions: List[Prediction]
    
