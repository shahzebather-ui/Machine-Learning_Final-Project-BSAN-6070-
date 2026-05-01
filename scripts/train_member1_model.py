from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COL = "target_hri_rate"
FEATURE_COLS = [
    "feat_mean_tmax_c_week",
    "feat_max_tmax_c_week",
    "feat_temp_range_c_week",
    "feat_heat_intensity",
    "feat_poverty_rate",
    "feat_unemployment_rate",
    "feat_median_hh_income",
    "feat_total_population",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [TARGET_COL, *FEATURE_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df.dropna(subset=required).copy()


def train_and_eval(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_rows_total": int(df.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "features": FEATURE_COLS,
        "target": TARGET_COL,
    }
    return model, metrics


def main() -> None:
    root = project_root()
    data_path = root / "data" / "dataset_finalized_region9_weekly_8features.csv"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    model, metrics = train_and_eval(df)

    model_path = models_dir / "member1_elasticnet.pkl"
    metrics_path = models_dir / "member1_metrics.json"
    sample_input_path = models_dir / "sample_input_for_prediction.csv"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    df[FEATURE_COLS].head(5).to_csv(sample_input_path, index=False)

    print("Training complete.")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Sample input saved: {sample_input_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
