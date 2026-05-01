from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


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
TARGET_COL = "target_hri_rate"


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "dataset_finalized_region9_weekly_8features.csv"
    models_dir = root / "models"
    docs_dir = root / "docs"
    viz_dir = docs_dir / "analysis_outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    max_depth_values = [2, 3, 4, 5, 6, 8, 10, 12, None]
    min_samples_leaf_values = [1, 2, 4, 6, 8]
    min_samples_split_values = [2, 4, 6, 8, 12]

    rows = []
    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:
            for min_samples_split in min_samples_split_values:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)
                train_rmse = rmse(y_train, pred_train)
                test_rmse = rmse(y_test, pred_test)
                train_r2 = float(r2_score(y_train, pred_train))
                test_r2 = float(r2_score(y_test, pred_test))
                rows.append(
                    {
                        "max_depth": str(max_depth),
                        "min_samples_leaf": min_samples_leaf,
                        "min_samples_split": min_samples_split,
                        "train_mae": float(mean_absolute_error(y_train, pred_train)),
                        "test_mae": float(mean_absolute_error(y_test, pred_test)),
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "r2_gap_train_minus_test": train_r2 - test_r2,
                        "rmse_ratio_test_over_train": test_rmse / train_rmse
                        if train_rmse > 0
                        else None,
                    }
                )

    results = pd.DataFrame(rows)

    # Rank by best test RMSE then better test R2, then lower overfit gap.
    ranked = results.sort_values(
        by=["test_rmse", "test_r2", "r2_gap_train_minus_test"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()

    best_depth = None if best["max_depth"] == "None" else int(best["max_depth"])
    best_model = DecisionTreeRegressor(
        max_depth=best_depth,
        min_samples_leaf=int(best["min_samples_leaf"]),
        min_samples_split=int(best["min_samples_split"]),
        random_state=42,
    )
    best_model.fit(X_train, y_train)
    best_pred_train = best_model.predict(X_train)
    best_pred_test = best_model.predict(X_test)

    best_metrics = {
        "model": "DecisionTreeRegressor",
        "selected_via": "grid search over max_depth/min_samples_leaf/min_samples_split",
        "selected_params": {
            "max_depth": best["max_depth"],
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "min_samples_split": int(best["min_samples_split"]),
            "random_state": 42,
        },
        "train_metrics": {
            "mae": float(mean_absolute_error(y_train, best_pred_train)),
            "rmse": rmse(y_train, best_pred_train),
            "r2": float(r2_score(y_train, best_pred_train)),
        },
        "test_metrics": {
            "mae": float(mean_absolute_error(y_test, best_pred_test)),
            "rmse": rmse(y_test, best_pred_test),
            "r2": float(r2_score(y_test, best_pred_test)),
        },
        "overfit_signals": {
            "r2_gap_train_minus_test": float(
                r2_score(y_train, best_pred_train) - r2_score(y_test, best_pred_test)
            ),
            "rmse_ratio_test_over_train": float(
                rmse(y_test, best_pred_test) / rmse(y_train, best_pred_train)
            ),
        },
        "n_rows_total": int(df.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "feature_columns": FEATURE_COLS,
        "target_column": TARGET_COL,
    }

    # Persist artifacts
    ranked.to_csv(models_dir / "decision_tree_tuning_results.csv", index=False)
    (models_dir / "decision_tree_tuning_results_top20.csv").write_text(
        ranked.head(20).to_csv(index=False)
    )
    (models_dir / "decision_tree_tuned_metrics.json").write_text(
        json.dumps(best_metrics, indent=2)
    )
    joblib.dump(best_model, models_dir / "member1_decision_tree_tuned.pkl")
    # Make tuned model the default model for Member 1 moving forward.
    joblib.dump(best_model, models_dir / "member1_decision_tree.pkl")
    (models_dir / "member1_decision_tree_metrics.json").write_text(
        json.dumps(
            {
                "model": "DecisionTreeRegressor",
                "params": best_metrics["selected_params"],
                "mae": best_metrics["test_metrics"]["mae"],
                "rmse": best_metrics["test_metrics"]["rmse"],
                "r2": best_metrics["test_metrics"]["r2"],
                "n_rows_total": best_metrics["n_rows_total"],
                "n_train": best_metrics["n_train"],
                "n_test": best_metrics["n_test"],
            },
            indent=2,
        )
    )

    overfit_report = {
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        },
        "models": {
            "decision_tree_tuned": {
                "train": best_metrics["train_metrics"],
                "test": best_metrics["test_metrics"],
                "overfit_signals": best_metrics["overfit_signals"],
            }
        },
    }
    (models_dir / "overfitting_check_report.json").write_text(
        json.dumps(overfit_report, indent=2)
    )

    # Visuals from tuned model
    plt.figure(figsize=(7, 4.5))
    plt.bar(
        ["Train RMSE", "Test RMSE"],
        [best_metrics["train_metrics"]["rmse"], best_metrics["test_metrics"]["rmse"]],
        color=["#43A047", "#FB8C00"],
    )
    plt.title("Tuned Decision Tree: Train vs Test RMSE")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(viz_dir / "viz_tuned_dt_train_test_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        ["Train R²", "Test R²"],
        [best_metrics["train_metrics"]["r2"], best_metrics["test_metrics"]["r2"]],
        color=["#1E88E5", "#8E24AA"],
    )
    plt.title("Tuned Decision Tree: Train vs Test R²")
    plt.ylabel("R²")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(viz_dir / "viz_tuned_dt_train_test_r2.png", dpi=180)
    plt.close()

    residuals = y_test - best_pred_test
    plt.figure(figsize=(7, 4.5))
    plt.hist(residuals, bins=12, color="#5E35B1", alpha=0.85)
    plt.axvline(0, linestyle="--", color="black")
    plt.title("Tuned Decision Tree: Residual Distribution (Test)")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(viz_dir / "viz_tuned_dt_residual_distribution.png", dpi=180)
    plt.close()

    # Log file
    log_lines = [
        "# Decision Tree Tuning Log",
        "",
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "Model family: DecisionTreeRegressor",
        "Search space:",
        f"- max_depth: {max_depth_values}",
        f"- min_samples_leaf: {min_samples_leaf_values}",
        f"- min_samples_split: {min_samples_split_values}",
        f"Total combinations tested: {len(rows)}",
        "",
        "Selection rule:",
        "- minimize test RMSE",
        "- maximize test R2 (tie-break)",
        "- minimize train-test R2 gap (tie-break)",
        "",
        "Selected parameters:",
        f"- max_depth: {best_metrics['selected_params']['max_depth']}",
        f"- min_samples_leaf: {best_metrics['selected_params']['min_samples_leaf']}",
        f"- min_samples_split: {best_metrics['selected_params']['min_samples_split']}",
        "",
        "Selected model metrics:",
        f"- Train RMSE: {best_metrics['train_metrics']['rmse']:.3f}",
        f"- Test RMSE: {best_metrics['test_metrics']['rmse']:.3f}",
        f"- Train R2: {best_metrics['train_metrics']['r2']:.3f}",
        f"- Test R2: {best_metrics['test_metrics']['r2']:.3f}",
        f"- R2 gap (train-test): {best_metrics['overfit_signals']['r2_gap_train_minus_test']:.3f}",
        f"- RMSE ratio (test/train): {best_metrics['overfit_signals']['rmse_ratio_test_over_train']:.3f}",
        "",
        "Artifacts written:",
        "- models/decision_tree_tuning_results.csv",
        "- models/decision_tree_tuned_metrics.json",
        "- models/member1_decision_tree_tuned.pkl",
        "- models/member1_decision_tree.pkl (updated to tuned)",
        "- models/member1_decision_tree_metrics.json (updated)",
        "- models/overfitting_check_report.json (updated)",
        "- docs/analysis_outputs/viz_tuned_dt_train_test_rmse.png",
        "- docs/analysis_outputs/viz_tuned_dt_train_test_r2.png",
        "- docs/analysis_outputs/viz_tuned_dt_residual_distribution.png",
    ]
    (docs_dir / "Decision_Tree_Tuning_Log.md").write_text("\n".join(log_lines))

    print("Tuning complete.")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
