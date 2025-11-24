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
def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")
    
    print("[OK] Esquema válido - Todas las columnas están presentes. ")

# Función 3: VaLidar los nulos
def validate_nulls(df: pd.DataFrame) -> None:
    null_ratios = df.isnull().mean()
    problematic = null_ratios[null_ratios > MAX_NULL_RATIO]
    
    if not problematic.empty:
        raise ValueError(
            "Hay columnas con demasiados valores de tipo Null:\n"
            f"{problematic}"
        )
    print("[OK] Porcentaje de valores de tipo Null aceptables en todas las columnas")

# Punto de entrada y orquestación del flujo de trabajo
def main() -> None:
    
    # 1. Carga de datos
    df = ingest_data(DATA_PATH)
    
    # 2. Validación del esquema
    validate_schema(df)
    
    # 3. Validación de datos Null
    validate_nulls(df)
    
    # 4. Definir ruta de salida para datos "validos"
    output_path = Path("data/churn_validated.csv")
    
    # 5. Guardar el DataFrame en un nuevo archivo CSV sin índice
    df.to_csv(output_path, index=False)
    
    # 6. Mensaje de finalización informando la ruta donde fueron guardado los datos validdos
    print(f"[GUARDADO] Datos validados guaddados en: {output_path}")

if __name__ == "__main__":
    main()