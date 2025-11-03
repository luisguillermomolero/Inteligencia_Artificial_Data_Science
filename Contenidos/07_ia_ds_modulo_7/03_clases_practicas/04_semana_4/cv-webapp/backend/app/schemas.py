from pydantic import BaseModel
from typing import Optional

class PredictResponse(BaseModel):
    class_id: int
    class_name: Optional[str] = None
    score: float

# No detection or base64 payloads needed anymore
