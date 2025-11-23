"""
Fase 2: Preprocesamiento y división train/test.

Aquí:
- Seleccionamos las columnas de entrada (features) y la columna objetivo.
- Escalamos las características numéricas con StandardScaler.
- Dividimos en train/test.
- Guardamos los datasets ya transformados y el preprocesador.

En MLOps, esta fase también se versiona (por ejemplo con DVC),
para saber con qué datos exactos entrenamos cada modelo.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

DATA_PATH = Path("data/churn_validated.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

FEATURE_COLS = [
    "age",
    "tenure",
    "monthly_fee",
    "num_products",
    "has_partner",
    "has_dependents",
]
TARGET_COL = "churn"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Datos validados cargados: {df.shape}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Reconstruimos DataFrames con nombres de columnas
    train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
    train_df[TARGET_COL] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    test_df[TARGET_COL] = y_test.reset_index(drop=True)

    # Guardamos
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[GUARDADO] Train guardado en: {TRAIN_PATH}")
    print(f"[GUARDADO] Test guardado en: {TEST_PATH}")
    print(f"[GUARDADO] Preprocesador guardado en: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
