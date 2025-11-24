import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Definir las rutas  los archivos
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
    "has_dependents"
]

TARGET_COL = "churn"

def main() -> None:
    
    # Cargar los datos
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Datos validados y cargados {df.shape}")
    
    # Separar los "features" (x)  "target" (y)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    # Escalado de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # División de TRAIN y TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Reconstrucción del DataFrame
    train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
    train_df[TARGET_COL] = y_train.reset_index(drop=True)
    
    test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    test_df[TARGET_COL] = y_test.reset_index(drop=True)
    
    # Guardar los archivos resultantes
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    
    test_df.to_csv(TEST_PATH, index=False)
    
    # Guardar el escalador p/reutilizar en producción
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"[GUARDADO] Train guardado en: {TRAIN_PATH}")
    print(f"[GUARDADO] Test guardado en: {TEST_PATH}")
    print(f"[GUARDADO] Preprocesador guardado en: {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    main()

