import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

TEST_PATH = Path("data/test.csv")

MODEL_PATH = Path("models/model.pkl")

FEATURES_COLS = [
    "age",
    "tenure",
    "monthly_fee",
    "num_products",
    "has_partner",
    "has_dependents",
]

TARGET_COL = "churn"

def main() -> None:
    
    df_test = pd.read_csv(TEST_PATH)
    
    X_test = df_test[FEATURES_COLS]
    y_test = df_test[TARGET_COL]
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    
    f1 = f1_score(y_test, y_pred)
    
    roc = roc_auc_score(y_test, y_proba)
    
    print(f"[OK] Test accuracy: {acc:.4f}")
    print(f"[OK] Test F1-score: {f1:.4f}")
    print(f"[OK] Test ROC-AUC:  {roc:.4f}")
    
    mlflow.set_experiment("churn_mlops_demo")
    
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc)

if __name__ == "__main__":
    main()