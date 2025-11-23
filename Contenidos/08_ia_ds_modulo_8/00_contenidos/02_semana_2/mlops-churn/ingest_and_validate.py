"""
Fase 1: Ingesta y validación de datos.

En esta etapa simulamos una validación mínima:
- Que existan todas las columnas esperadas.
- Que no haya columnas con demasiados valores nulos.

En un MLOps real, esta validación podría ser orquestada por Airflow
y apoyarse en herramientas de validación de datos como Great Expectations.
"""

import pandas as pd
from pathlib import Path

# Ruta del dataset original
DATA_PATH = Path("data/churn_data.csv")

# Esquema esperado
EXPECTED_COLUMNS = [
    "age",
    "tenure",
    "monthly_fee",
    "num_products",
    "has_partner",
    "has_dependents",
    "churn",
]

# Porcentaje máximo de nulos permitido por columna
MAX_NULL_RATIO = 0.3


def ingest_data(path: Path) -> pd.DataFrame:
    """Carga el dataset desde disco."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    print(f"[CARGA] Datos cargados con shape: {df.shape}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Valida que existan todas las columnas esperadas."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")
    print("[OK] Esquema validado (todas las columnas estan presentes).")


def validate_nulls(df: pd.DataFrame) -> None:
    """Valida que no haya muchas columnas con nulos."""
    null_ratios = df.isnull().mean()
    problematic = null_ratios[null_ratios > MAX_NULL_RATIO]
    if not problematic.empty:
        raise ValueError(
            "Hay columnas con demasiados valores nulos:\n"
            f"{problematic}"
        )
    print("[OK] Porcentaje de nulos aceptable en todas las columnas.")


def main() -> None:
    df = ingest_data(DATA_PATH)
    validate_schema(df)
    validate_nulls(df)

    # Guardamos la versión validada
    output_path = Path("data/churn_validated.csv")
    df.to_csv(output_path, index=False)
    print(f"[GUARDADO] Datos validados guardados en: {output_path}")


if __name__ == "__main__":
    main()
