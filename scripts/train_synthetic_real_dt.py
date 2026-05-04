"""
Train a DecisionTreeRegressor on synthetic data only; evaluate on real data only.

Data roles (Step 1 in the project workflow):
  - Fit:  data/synthetic_hri_dataset_fixed.csv
  - Test: data/final_hri_modeling_dataset.csv

The real file may include `regionid`; it is not used as a feature unless you pass
--include-regionid (usually leave it out).

Saves (defaults; override paths so ablations do not overwrite your main artifact):
  - models/member1_decision_tree.pkl
  - models/member1_decision_tree_metrics.json
  - models/member1_training_runs_log.csv   ← appends one row per run (settings + scores)

Run from project root:
  python3 scripts/train_synthetic_real_dt.py --max-depth 8 --min-samples-leaf 10

Feature ablation without touching the default .pkl (example):
  python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 \\
    --min-samples-split 2 --exclude-features feat_unemployment_rate \\
    --model-out models/backup/member1_ablation_no_unemployment.pkl \\
    --metrics-out models/backup/member1_ablation_no_unemployment_metrics.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

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


def _append_run_log(log_path: Path, log_row: dict) -> None:
    """Append one row; extend columns if new fields appear (older rows get NaN for new cols)."""
    new = pd.DataFrame([log_row])
    if log_path.exists() and log_path.stat().st_size > 0:
        old = pd.read_csv(log_path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(log_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decision Tree: synthetic train → real test.")
    p.add_argument(
        "--synthetic",
        default="data/synthetic_hri_dataset_fixed.csv",
        help="Synthetic training CSV path (relative to repo root).",
    )
    p.add_argument(
        "--real",
        default="data/final_hri_modeling_dataset.csv",
        help="Real holdout CSV path (relative to repo root).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--include-regionid",
        action="store_true",
        help="If set, includes regionid as a numeric feature (default: off).",
    )
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--min-samples-split", type=int, default=2)
    p.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        metavar="COL",
        help="Feature column names to remove for this run (ablation / feature drop).",
    )
    p.add_argument(
        "--model-out",
        default="models/member1_decision_tree.pkl",
        help="Path for saved .pkl relative to repo root (use e.g. models/backup/... for ablations).",
    )
    p.add_argument(
        "--metrics-out",
        default="models/member1_decision_tree_metrics.json",
        help="Path for metrics JSON relative to repo root (must pair with the same run as --model-out).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

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

    # Drop rows with NaNs in modeling columns (should be none if data is clean)
    train_ok = X_train.notna().all(axis=1) & y_train.notna()
    test_ok = X_test.notna().all(axis=1) & y_test.notna()
    X_train, y_train = X_train.loc[train_ok], y_train.loc[train_ok]
    X_test, y_test = X_test.loc[test_ok], y_test.loc[test_ok]

    model = DecisionTreeRegressor(
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(a, b)))

    metrics = {
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
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "min_samples_split": args.min_samples_split,
            "random_state": args.random_state,
            "excluded_features": list(args.exclude_features) if args.exclude_features else [],
        },
        "feature_importances": {
            c: float(v) for c, v in zip(feature_cols, model.feature_importances_)
        },
    }

    # Streamlit card reads top-level mae / rmse / r2 → use **real test** here
    metrics["mae"] = metrics["test_on_real"]["mae"]
    metrics["rmse"] = metrics["test_on_real"]["rmse"]
    metrics["r2"] = metrics["test_on_real"]["r2"]

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = root / args.model_out
    metrics_path = root / args.metrics_out
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_path = models_dir / "member1_training_runs_log.csv"
    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "min_samples_split": args.min_samples_split,
        "random_state": args.random_state,
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

    rel_model = model_path.relative_to(root)
    rel_metrics = metrics_path.relative_to(root)
    print()
    print("========== WHERE FILES ARE (project folder) ==========")
    print(f"  Model:  {rel_model}")
    print(f"  Metrics JSON: {rel_metrics}")
    print("  Run log (all tries): models/member1_training_runs_log.csv")
    print()
    print("========== THIS RUN — SETTINGS YOU USED ==========")
    print(
        f"  max_depth={args.max_depth!r}  min_samples_leaf={args.min_samples_leaf}  min_samples_split={args.min_samples_split}"
    )
    if args.exclude_features:
        print(f"  EXCLUDED features: {', '.join(args.exclude_features)}")
    print(f"  Features used ({len(feature_cols)}): {', '.join(feature_cols)}")
    print()
    print("========== THIS RUN — SCORES (real 1460 rows) ==========")
    print("  MAE :", round(metrics["test_on_real"]["mae"], 4))
    print("  RMSE:", round(metrics["test_on_real"]["rmse"], 4))
    print("  R2  :", round(metrics["test_on_real"]["r2"], 4))
    print()
    print("========== THIS RUN — SCORES (synthetic train, fit set) ==========")
    print("  MAE :", round(metrics["train_on_synthetic"]["mae"], 4))
    print("  RMSE:", round(metrics["train_on_synthetic"]["rmse"], 4))
    print("  R2  :", round(metrics["train_on_synthetic"]["r2"], 4))
    print()
    print("Also appended one row to:", log_path.name)


if __name__ == "__main__":
    main()
