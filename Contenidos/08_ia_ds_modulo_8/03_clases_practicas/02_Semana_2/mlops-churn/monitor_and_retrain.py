import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn
import pickle

mlflow.set_tracking_uri("file:./mlruns")

NEW_DATA_PATH = Path("data/new_churn_data.csv")

TRAIN_PATH = Path("data/train.csv")

MODEL_PATH = Path("models/model.pkl")

PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

FEATURE_COLS = [
    "age",
    "tenure",
    "monthly_fee",
    "num_products"
    "has_partner",
    "has_dependents",
]

TARGET_COL = "churn"

PERFORMANCE_THRESHOLD = 0.80

def main() -> None:
    
    if not NEW_DATA_PATH.exists():
        print("[ADVERTENCIA] No hay datos nuevos para monitorizar")
        return
    
    df_new_raw = pd.read_csv(NEW_DATA_PATH)
    
    with open(PREPROCESSOR_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    X_new = df_new_raw[FEATURE_COLS]
    y_new = df_new_raw[TARGET_COL]
    
    X_new_scaled = scaler.transform(X_new)
    
    with open(MODEL_PATH, "rb") as f:
        current_model = pickle.load(f)
        
    y_pred_new = current_model.predict(X_new_scaled)
    acc_new = accuracy_score(y_new, y_pred_new)
    
    print(f"[INFO] Accuracy en datos nuevo: {acc_new:.4f}")
    
    mlflow.set_experiment("churn_mlops_demo")
    with mlflow.start_run(run_name="monitoring"):
        mlflow.log_metric("new_data_accuracy", acc_new)
        
    if acc_new < PERFORMANCE_THRESHOLD:
        print("[ADVERTENCIA] Rendimiento bajo, iniciando reentrenamiento...")
    
        df_train_scaled = pd.read_csv(TRAIN_PATH)
        
        df_new_scaled = pd.DataFrame(X_new_scaled, columns=FEATURE_COLS)
        df_new_scaled = [TARGET_COL] = y_new.reset_index(drop=True)
        
        df_full = pd.concat (
            [df_train_scaled, df_new_scaled],
            ignore_index=True
        )
        
        x_full = df_full[FEATURE_COLS]
        y_full = df_full[TARGET_COL]
        
        X_train, X_val, y_train, y_val = train_test_split(
            x_full,
            y_full,
            test_size=0.2,
            random_state=42,
            stratify=y_full
        )
        
        new_model = LogisticRegression(
            max_iter=300,
            solver="liblinear"
        )
        
        new_model.fit(X_train, y_train)
        
        acc_val = accuracy_score(y_val, new_model.predict(X_val))
        print(f"[OK] Nuevo modelo entrenado. Accuracy validation: {acc_val:.4f}")
        
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)
        
        with mlflow.start_run(run_name="retrain"):
            mlflow.log_metric("retrain_val_accuracy", acc_val)
            
            signature = infer_signature(X_train, new_model.predict(X_train))
            input_example = X_train.head(1)
            
            mlflow.sklearn.log_model(
                sk_model=new_model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="churn_model"
            )
        
        print("[GUARDADO] Modelo actualizado guardado en disco y registrado en MLflow")
    
    else:
        print("[OK] El modelo mantiene un rendimiento aceptable. No se reentrena.")

if __name__ == "__main__":
    main()