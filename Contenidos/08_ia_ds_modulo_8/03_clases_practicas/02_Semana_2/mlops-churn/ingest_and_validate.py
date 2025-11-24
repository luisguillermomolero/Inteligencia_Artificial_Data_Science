import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/churn_data.csv")

EXPECTED_COLUMNS = [
    "age",
    "tenure",
    "monthly_fee",
    "num_products",
    "has_partner",
    "has_dependents",
    "churn"
]

MAX_NULL_RATIO = 0.3

# Función 1: Ingesta de datos
def ingest_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")
    
    df = pd.read_csv(path)
    print(f"[CARGA] Datos cargados con shape: {df.shape}")
    return df

# Función 2: validación del esquema

