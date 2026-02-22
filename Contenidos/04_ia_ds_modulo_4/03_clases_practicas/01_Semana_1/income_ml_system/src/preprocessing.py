import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def build_preprocessor() -> ColumnTransformer:
    
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor

def save_preprocessor(preprocessor):
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
    
