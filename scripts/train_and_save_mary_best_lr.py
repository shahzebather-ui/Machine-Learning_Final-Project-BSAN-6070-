"""
Train Mary's engineered LinearRegression (synthetic train / real test) and save artifacts.

Matches notebook logic in:
  Best Model/Linear Regression - Mary-selected/Final_Linear_Regression (1).ipynb

Reads:
  data/synthetic_hri_dataset_fixed.csv  (project canonical synthetic CSV)
  data/final_hri_modeling_dataset.csv

Writes:
  models/mary_best_engineered_linear_regression.joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["min_temp_sq"] = out["min_temp_celsius"] ** 2
    out["log_unsheltered_homeless"] = np.log1p(out["unsheltered_homeless"])
    out["temp_income_interaction"] = out["min_temp_celsius"] * out["feat_median_hh_income"]
    return out


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    synth = root / "data/synthetic_hri_dataset_fixed.csv"
    real = root / "data/final_hri_modeling_dataset.csv"
    out_model = root / "models/mary_best_engineered_linear_regression.joblib"
    verify_csv = (
        root
        / "Best Model/Linear Regression - Mary-selected/sample_input_output.csv"
    )

    train_df = add_engineered_features(pd.read_csv(synth))
    test_df = add_engineered_features(pd.read_csv(real))

    target = "hri_value"
    feature_cols = [
        "min_temp_celsius",
        "min_temp_sq",
        "feat_median_hh_income",
        "log_unsheltered_homeless",
        "temp_income_interaction",
    ]

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[target].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[target].astype(float)

    tr_ok = X_train.notna().all(axis=1) & y_train.notna()
    te_ok = X_test.notna().all(axis=1) & y_test.notna()
    X_train, y_train = X_train.loc[tr_ok], y_train.loc[tr_ok]
    X_test, y_test = X_test.loc[te_ok], y_test.loc[te_ok]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "model_name": "Engineered Linear Regression",
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": rmse(y_test.values, pred),
        "r2": float(r2_score(y_test, pred)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols": feature_cols,
        "train_csv": str(synth.relative_to(root)),
        "test_csv": str(real.relative_to(root)),
    }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)
    (root / "models/mary_best_engineered_linear_regression_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Saved model:", out_model.relative_to(root))
    print("Metrics:", json.dumps(metrics, indent=2)[:800])

    if verify_csv.exists():
        row = pd.read_csv(verify_csv).iloc[0]
        Xv = pd.DataFrame([row[feature_cols].astype(float).values], columns=feature_cols)
        pv = float(model.predict(Xv)[0])
        exp = float(row["expected_prediction"])
        diff = abs(pv - exp)
        print("\nSample verify:")
        print("  predict:", pv)
        print("  expected:", exp)
        print("  abs diff:", diff)
        if diff > 1e-3:
            raise SystemExit(
                "Sample prediction mismatch — check CSV/engineering vs training data."
            )


if __name__ == "__main__":
    main()
