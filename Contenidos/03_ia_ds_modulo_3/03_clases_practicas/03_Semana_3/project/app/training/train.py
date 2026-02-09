from __future__ import annotations
from math import sqrt
from pathlib import Path
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from pipeline.pipeline import (build_pipeline, get_feature_columns, load_dataset, split_features_target)

def train_and_evaluate(model_path: Path) -> None:
    df = load_dataset()
    features, target = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    pipeline = build_pipeline(get_feature_columns())
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = _compute_rmse(y_test, predictions, mse)
    r2 = r2_score(y_test, predictions)
    
    print("Métricas de evaluación en conjunto de prueba (test):")
    print(f"- MAE  (Error absoluto medio)        : {mae:.4f}    (menor es mejor)")
    print(f"- MSE  (Error cuadrático medio)      : {mse:.4f}    (menor es mejor)")
    print(f"- RMSE (Raíz del MSE)                : {rmse:.4f}   (menor es mejor)")
    print(f"- R2 (Coeficiente de determinación)  : {r2:.4}      (más cerca a 1 es mejor)")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline guardado en: {model_path}")


def _compute_rmse(y_true, y_pred, mse_value: float) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        return float(sqrt(mse_value))
    
def main() -> None:
    model_path = Path("model") / "pipeline.joblib"
    train_and_evaluate(model_path)

if __name__ == "__main__":
    main()
    