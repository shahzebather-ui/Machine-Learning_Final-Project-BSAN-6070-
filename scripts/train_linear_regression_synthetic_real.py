"""
LinearRegression (OLS): synthetic train -> real test.

Mary (Member 2) pipeline. Same data rules as Decision Tree training.

Run from repo root:
  python3 scripts/train_linear_regression_synthetic_real.py
  python3 scripts/train_linear_regression_synthetic_real.py \\
    --exclude-features max_temp_celsius feat_unemployment_rate feat_poverty_rate overall_homeless \\
    --model-out models/mary_linear_regression_4feat.pkl \\
    --metrics-out models/mary_linear_regression_4feat_metrics.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COL = "hri_value"
BASE_FEATURES = [
    "max_temp_celsius",
    "min_temp_celsius",
    "feat_poverty_rate",
    "feat_unemployment_rate",
    "feat_median_hh_income",
    "feat_total_population",
    "overall_homeless",
    "unsheltered_homeless",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _append_run_log(log_path: Path, log_row: dict) -> None:
    new = pd.DataFrame([log_row])
    if log_path.exists() and log_path.stat().st_size > 0:
        old = pd.read_csv(log_path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(log_path, index=False)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Linear regression: synthetic train -> real test.")
    p.add_argument("--synthetic", default="data/synthetic_hri_dataset_fixed.csv")
    p.add_argument("--real", default="data/final_hri_modeling_dataset.csv")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--include-regionid", action="store_true")
    p.add_argument("--exclude-features", nargs="*", default=[], metavar="COL")
    p.add_argument(
        "--model-out",
        default="models/mary_linear_regression_8feat.pkl",
        help="Path relative to repo root.",
    )
    p.add_argument(
        "--metrics-out",
        default="models/mary_linear_regression_8feat_metrics.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()

    synth_path = root / args.synthetic
    real_path = root / args.real

    feature_cols = list(BASE_FEATURES)
    if args.include_regionid:
        feature_cols = ["regionid", *feature_cols]
    if args.exclude_features:
        bad = set(args.exclude_features) - set(feature_cols)
        if bad:
            raise ValueError(f"--exclude-features unknown: {sorted(bad)}")
        feature_cols = [c for c in feature_cols if c not in set(args.exclude_features)]

    train_df = pd.read_csv(synth_path)
    real_df = pd.read_csv(real_path)

    for name, df in [("synthetic", train_df), ("real", real_df)]:
        missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].astype(float)
    X_test = real_df[feature_cols].copy()
    y_test = real_df[TARGET_COL].astype(float)

    train_ok = X_train.notna().all(axis=1) & y_train.notna()
    test_ok = X_test.notna().all(axis=1) & y_test.notna()
    X_train, y_train = X_train.loc[train_ok], y_train.loc[train_ok]
    X_test, y_test = X_test.loc[test_ok], y_test.loc[test_ok]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(a, b)))

    coefs = {c: float(v) for c, v in zip(feature_cols, model.coef_)}

    metrics = {
        "model_type": "LinearRegression",
        "target_col": TARGET_COL,
        "feature_cols": feature_cols,
        "n_train_synthetic": int(len(X_train)),
        "n_test_real": int(len(X_test)),
        "train_on_synthetic": {
            "mae": float(mean_absolute_error(y_train, y_pred_train)),
            "rmse": rmse(y_train.values, y_pred_train),
            "r2": float(r2_score(y_train, y_pred_train)),
        },
        "test_on_real": {
            "mae": float(mean_absolute_error(y_test, y_pred_test)),
            "rmse": rmse(y_test.values, y_pred_test),
            "r2": float(r2_score(y_test, y_pred_test)),
        },
        "model_params": {
            "random_state": args.random_state,
            "include_regionid": args.include_regionid,
            "excluded_features": list(args.exclude_features) if args.exclude_features else [],
        },
        "intercept": float(model.intercept_),
        "coefficients": coefs,
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "rmse": float(rmse(y_test.values, y_pred_test)),
        "r2": float(r2_score(y_test, y_pred_test)),
    }

    model_path = root / args.model_out
    metrics_path = root / args.metrics_out
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_path = root / "models" / "mary_lr_training_runs_log.csv"
    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "n_features": len(feature_cols),
        "excluded_features": ",".join(args.exclude_features) if args.exclude_features else "",
        "real_test_rmse": round(metrics["test_on_real"]["rmse"], 6),
        "real_test_mae": round(metrics["test_on_real"]["mae"], 6),
        "real_test_r2": round(metrics["test_on_real"]["r2"], 6),
    }
    _append_run_log(log_path, log_row)

    print("Saved:", model_path.relative_to(root))
    print("Saved:", metrics_path.relative_to(root))
    print("Real holdout RMSE:", round(metrics["rmse"], 4))


if __name__ == "__main__":
    main()
