"""
Grid search over **every non-empty subset** of the 8 default numeric features.

Workflow matches the course setup:
  - **Train / fit:** synthetic CSV only
  - **Evaluate:** real holdout CSV only

Default tree hyperparameters match the frozen Member 1 DT unless you override flags:
  max_depth=5, min_samples_leaf=9, min_samples_split=2, random_state=42

Combinatorics:
  - 8 features → 2^8 subsets → **256** total, minus **1** empty set → **255** models.

Outputs:
  - Prints a ranked table (stdout)
  - Writes **only the top N rows** (default 15) to CSV: models/dt_feature_subset_top15_lowest_rmse.csv
  - Use --full-results-out PATH if you explicitly want **all** subset rows saved again.

Usage (from repo root):
  python3 scripts/grid_dt_all_feature_subsets.py

Optional:
  python3 scripts/grid_dt_all_feature_subsets.py --save-rows 10
  python3 scripts/grid_dt_all_feature_subsets.py --full-results-out models/dt_feature_subset_grid_results_FULL.csv
"""

from __future__ import annotations

import argparse
from itertools import chain, combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

TARGET_COL = "hri_value"

# Same 8 numeric inputs as scripts/train_synthetic_real_dt.py (order preserved).
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


def _powerset(features: list[str]):
    """Yield every subset as a tuple, including empty — caller should skip empty."""
    return chain.from_iterable(combinations(features, r) for r in range(len(features) + 1))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Train DT on every non-empty feature subset; score on real holdout."
    )
    p.add_argument("--synthetic", default="data/synthetic_hri_dataset_fixed.csv")
    p.add_argument("--real", default="data/final_hri_modeling_dataset.csv")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-samples-leaf", type=int, default=9)
    p.add_argument("--min-samples-split", type=int, default=2)
    p.add_argument(
        "--sort-by",
        choices=("rmse", "mae", "r2"),
        default="rmse",
        help="Rank rows: rmse/mae ascending (lower better), r2 descending (higher better).",
    )
    p.add_argument("--top", type=int, default=30, help="How many rows to print.")
    p.add_argument(
        "--out-csv",
        default="models/dt_feature_subset_top15_lowest_rmse.csv",
        help="Where to save the top subset rows after ranking (relative to repo root).",
    )
    p.add_argument(
        "--save-rows",
        type=int,
        default=15,
        help="How many top rows (after --sort-by) to write to --out-csv.",
    )
    p.add_argument(
        "--full-results-out",
        default=None,
        help="If set, also write ALL ranked rows to this path (large file). Omit to avoid that.",
    )
    args = p.parse_args()

    synth_path = root / args.synthetic
    real_path = root / args.real
    train_df = pd.read_csv(synth_path)
    real_df = pd.read_csv(real_path)

    rows: list[dict] = []

    # --- Enumerate all non-empty feature subsets ---
    for subset in _powerset(BASE_FEATURES):
        if len(subset) == 0:
            continue

        feature_cols = list(subset)

        # Sanity: both CSVs must contain these columns + target
        missing_s = [c for c in feature_cols + [TARGET_COL] if c not in train_df.columns]
        missing_r = [c for c in feature_cols + [TARGET_COL] if c not in real_df.columns]
        if missing_s or missing_r:
            raise ValueError(f"Missing columns synthetic={missing_s} real={missing_r}")

        X_train = train_df[feature_cols].copy()
        y_train = train_df[TARGET_COL].astype(float)
        X_test = real_df[feature_cols].copy()
        y_test = real_df[TARGET_COL].astype(float)

        # Drop incomplete rows (should be rare if data are clean)
        tr_ok = X_train.notna().all(axis=1) & y_train.notna()
        te_ok = X_test.notna().all(axis=1) & y_test.notna()
        X_train, y_train = X_train.loc[tr_ok], y_train.loc[tr_ok]
        X_test, y_test = X_test.loc[te_ok], y_test.loc[te_ok]

        # --- Same estimator settings as your frozen Member 1 tree (unless overridden) ---
        model = DecisionTreeRegressor(
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
        )
        model.fit(X_train, y_train)

        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)

        rows.append(
            {
                "features": "|".join(feature_cols),
                "n_features": len(feature_cols),
                "real_mae": float(mean_absolute_error(y_test, pred_te)),
                "real_rmse": _rmse(y_test.values, pred_te),
                "real_r2": float(r2_score(y_test, pred_te)),
                "synth_rmse_insample": _rmse(y_train.values, pred_tr),
            }
        )

    results = pd.DataFrame(rows)

    # --- Rank for reporting ---
    ascending = args.sort_by != "r2"
    results_sorted = results.sort_values(
        by=f"real_{args.sort_by}" if args.sort_by != "r2" else "real_r2",
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)

    out_path = root / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_to_save = results_sorted.head(args.save_rows).reset_index(drop=True)
    top_to_save.insert(0, "rank", range(1, len(top_to_save) + 1))
    top_to_save.to_csv(out_path, index=False)

    if args.full_results_out:
        full_path = root / args.full_results_out
        full_path.parent.mkdir(parents=True, exist_ok=True)
        results_sorted.to_csv(full_path, index=False)

    print(f"Trained {len(results)} models (all non-empty subsets of {len(BASE_FEATURES)} features).")
    print(f"Hyperparameters: max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}, ")
    print(f"                 min_samples_split={args.min_samples_split}, random_state={args.random_state}")
    print(
        f"Saved top {len(top_to_save)} rows CSV: {out_path.relative_to(root)} "
        f"(sort key: real_{args.sort_by})."
    )
    if args.full_results_out:
        print(f"Also saved FULL ranked CSV: {(root / args.full_results_out).relative_to(root)}")
    print()
    print(f"=== Top {args.top} by real_{args.sort_by} ({'lower' if ascending else 'higher'} is better) ===")

    disp = results_sorted.head(args.top).copy()
    # Widen readable feature list for terminal / notebook
    disp["features_csv"] = disp["features"].str.replace("|", ", ")
    cols_show = ["n_features", "real_rmse", "real_mae", "real_r2", "features_csv"]
    # Format floats for display-only copy
    fmt_disp = disp[cols_show].copy()
    fmt_disp["real_rmse"] = fmt_disp["real_rmse"].map(lambda x: f"{x:.4f}")
    fmt_disp["real_mae"] = fmt_disp["real_mae"].map(lambda x: f"{x:.4f}")
    fmt_disp["real_r2"] = fmt_disp["real_r2"].map(lambda x: f"{x:.4f}")

    print(fmt_disp.to_string(index=False))


if __name__ == "__main__":
    main()
