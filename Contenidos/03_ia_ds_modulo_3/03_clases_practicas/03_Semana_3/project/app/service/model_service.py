from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from pipeline.pipeline import get_feature_columns

class ModelNotLoadedError(RuntimeError):
    pass

@dataclass
class ModelService:
    model_path: Path
    pipeline: Pipeline
    
    @classmethod
    def load(cls, model_path: Path) -> "ModelService":
        if not model_path.exists():
            raise ModelNotLoadedError("Modelo no disponible")
        
        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            raise ModelNotLoadedError("Modelo no disponible")
        return cls(
            model_path=model_path,
            pipeline=pipeline,
        )
    
    def predict(self, features: Dict[str, float]) -> float:
        feature_columns = get_feature_columns()
        data = pd.DataFrame(
            [features],
            columns=feature_columns,
        )
        prediction = self.pipeline.predict(data)[0]
        
        return float(prediction)
