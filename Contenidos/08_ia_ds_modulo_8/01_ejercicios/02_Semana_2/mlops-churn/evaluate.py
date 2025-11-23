"""
Fase 4: Evaluación del modelo.

Aquí:
- Cargamos el modelo entrenado.
- Evaluamos en el conjunto de prueba.
- Registramos métricas en MLflow.

Esta evaluación es clave en MLOps para decidir si un modelo
es lo suficientemente bueno para ir a producción.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
import mlflow

# Configurar MLflow para usar SQLite en lugar de filesystem
mlflow.set_tracking_uri("sqlite:///mlflow.db")

TEST_PATH = Path("data/test.csv")
MODEL_PATH = Path("models/model.pkl")

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
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"[OK] Test Accuracy: {acc:.4f}")
    print(f"[OK] Test F1-score: {f1:.4f}")
    print(f"[OK] Test ROC-AUC:  {roc:.4f}")

    mlflow.set_experiment("churn_mlops_demo")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc)


if __name__ == "__main__":
    main()
