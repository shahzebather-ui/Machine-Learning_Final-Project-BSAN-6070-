"""
LinearRegression (OLS): synthetic train → real test.

Isolated from the Decision Tree pipeline: default outputs live under
  experiments/linear_regression/outputs/
so main Member 1 artifacts in models/ are never overwritten.

Run from repo root:
  python3 experiments/linear_regression/train_linear_regression_synthetic_real.py
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

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
DEFAULT_MODEL_OUT = OUT_DIR / "linear_regression_8feat.pkl"
DEFAULT_METRICS_OUT = OUT_DIR / "linear_regression_8feat_metrics.json"
LOG_NAME = "lr_training_runs_log.csv"


def _append_run_log(log_path: Path, log_row: dict) -> None:
    new = pd.DataFrame([log_row])
    if log_path.exists() and log_path.stat().st_size > 0:
        old = pd.read_csv(log_path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(log_path, index=False)


def _repo_root() -> Path:
    # experiments/linear_regression/this_file.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear regression: synthetic train → real test.")
    p.add_argument(
        "--synthetic",
        default="data/synthetic_hri_dataset_fixed.csv",
        help="Synthetic training CSV (relative to repo root).",
    )
    p.add_argument(
        "--real",
        default="data/final_hri_modeling_dataset.csv",
        help="Real holdout CSV (relative to repo root).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--include-regionid",
        action="store_true",
        help="If set, prepend regionid as a numeric feature.",
    )
    p.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        metavar="COL",
        help="Feature columns to drop for this run (ablation).",
    )
    p.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_OUT.relative_to(_repo_root())),
        help="Model .pkl path relative to repo root (keep under experiments/linear_regression/outputs/).",
    )
    p.add_argument(
        "--metrics-out",
        default=str(DEFAULT_METRICS_OUT.relative_to(_repo_root())),
        help="Metrics JSON path relative to repo root.",
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
            raise ValueError(f"--exclude-features unknown or not in current set: {sorted(bad)}")
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
    }

    metrics["mae"] = metrics["test_on_real"]["mae"]
    metrics["rmse"] = metrics["test_on_real"]["rmse"]
    metrics["r2"] = metrics["test_on_real"]["r2"]

    model_path = root / args.model_out
    metrics_path = root / args.metrics_out
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_path = OUT_DIR / LOG_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "LinearRegression",
        "include_regionid": args.include_regionid,
        "n_features": len(feature_cols),
        "excluded_features": ",".join(args.exclude_features) if args.exclude_features else "",
        "n_synthetic_train": len(X_train),
        "n_real_test": len(X_test),
        "real_test_mae": round(metrics["test_on_real"]["mae"], 6),
        "real_test_rmse": round(metrics["test_on_real"]["rmse"], 6),
        "real_test_r2": round(metrics["test_on_real"]["r2"], 6),
        "synthetic_train_rmse": round(metrics["train_on_synthetic"]["rmse"], 6),
    }
    _append_run_log(log_path, log_row)

    print()
    print("========== LINEAR REGRESSION (isolated experiment) ==========")
    print(f"  Model:  {model_path.relative_to(root)}")
    print(f"  Metrics: {metrics_path.relative_to(root)}")
    print(f"  Run log: {log_path.relative_to(root)}")
    print()
    print(f"  Features used ({len(feature_cols)}): {', '.join(feature_cols)}")
    print()
    print("========== SCORES (real holdout) ==========")
    print("  MAE :", round(metrics["test_on_real"]["mae"], 4))
    print("  RMSE:", round(metrics["test_on_real"]["rmse"], 4))
    print("  R2  :", round(metrics["test_on_real"]["r2"], 4))
    print()
    print("========== SCORES (synthetic train, in-sample) ==========")
    print("  MAE :", round(metrics["train_on_synthetic"]["mae"], 4))
    print("  RMSE:", round(metrics["train_on_synthetic"]["rmse"], 4))
    print("  R2  :", round(metrics["train_on_synthetic"]["r2"], 4))
    print()
    print("Coefficients (see metrics JSON):", ", ".join(f"{k}={v:.6g}" for k, v in coefs.items()))


if __name__ == "__main__":
    main()
