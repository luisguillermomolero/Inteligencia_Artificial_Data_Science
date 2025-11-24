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
        
                