from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_dataset_url() -> str:
    return (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00374/energydata_complete.csv"
    )

def get_cache_path() -> Path:
    return Path("data") / "energydata_complete.csv"

def get_feature_columns() -> List[str]:
    return ["T1", "RH_1", "T_out", "Windspeed"]

def get_target_column() -> str:
    return "Appliances"

def download_dataset(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    
    urlretrieve(url, cache_path.as_posix())
    
    return cache_path

def load_dataset() -> pd.DataFrame:
    dataset_path = download_dataset(
        get_dataset_url(),
        get_cache_path()
    )
    
    df = pd.read_csv(dataset_path)
    
    columns = get_feature_columns() + [get_target_column()]
    
    return df[columns].copy()

def build_pipeline(feature_columns: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                feature_columns
            )
        ]
    )
    
    model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42
    )
    
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = get_feature_columns()
    target_columns = get_target_column()
    return df[feature_columns], df[target_columns]

