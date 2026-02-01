from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Declarar constantes globales

# Data URL
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

# Función de carga y caching dee datos
def load_data(
    data_url: str = DATA_URL, 
    cache_path: Path = Path("data/winequality-red.csv")
    ) -> pd.DataFrame:
    
    if cache_path.exists():
        return pd.read_csv(cache_path, sep=";")
    
    df = pd.read_csv(data_url, sep=";")
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(cache_path, index=False, sep=";")
    
    return df

# Función de separación de features(caracteristicas, Xn, ó...) / el target(y, etiqueta ó...)
def split_features_target(
    df: pd.DataFrame, 
    target: str = "quality"
    ) -> Tuple[pd.DataFrame, pd.Series]:
    
    x = df.drop(columns=[target])
    
    y = df[target]
    
    return x, y


# Función de construcción del pipeline
def built_pipeline(numeric_features: list[str]) -> Pipeline:
    
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer", 
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            )   
        ],
        remainder="drop"
    )
    
    model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )
    
    return pipeline