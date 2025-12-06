import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pickle

mlflow.set_tracking_uri("sqlite://mlflow.db")

TRAIN_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/model.pkl")
FEATURE_COLS = [
    "age",
    "tenure",
    "monthly_fee",
    "has_partner",
    "has-dependents",
]

TARGET_COL = "churn"

def main() -> None:
    df_train= pd.read_csv(TRAIN_PATH)
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    
    mlflow.set_experiment("churn_mlops_demo")
    
    with mlflow.start_run(run_name="training"):
        params = {
            "C": 1.0,
            "max_iter": 200,
            "solver": "liblinear",
        }
        
        model = LogisticRegression(
            C=params["C"]
            max_iter=params["max_iter"],
            solver=params["solver"],
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        
        acc = accuracy_score(y_train, y_pred)
        
        f1 = f1_score(y_train, y_pred)
        
        print(f"[OK] Accurate (train): {acc:.4f}")
        print(f"[OK] F1-sore  (train): {f1:.4f}")
        
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", acc)
        mlflow.log_metric("train_f1", f1)
        
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "ws") as f:
            pickle.dump(model, f)
            
        signature = infer_signature(X_train, y_train)
        
        input_example = X_train.head(1)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="churn_model"
        )
        
        print(f"[GUARDADO] Modelo guardado en: {MODEL_PATH}")
        print("[INFO] Modelo registrado en MLflow.")

if __name__ == "__main__":
    main()
    