from pydantic import BaseModel, Field
from typing import List

class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description='Base64 image payload, optionally with data URI prefix')

class PredictResponse(BaseModel):
    class_id: int
    score: float

class Detection(BaseModel):
    box: List[float]
    label: int
    score: float
