from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "dataset_finalized_region9_weekly_8features.csv"
    model_path = root / "models" / "member1_elasticnet.pkl"
    out_dir = root / "docs" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).dropna(subset=[TARGET_COL, *FEATURE_COLS]).copy()

    # 1) Correlation heatmap
    corr_cols = [TARGET_COL, *FEATURE_COLS]
    corr = df[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation Heatmap: Target vs Final Features")
    plt.tight_layout()
    plt.savefig(out_dir / "viz_correlation_heatmap.png", dpi=180)
    plt.close()

    # 2) Target trend by year (line)
    if "record_year" in df.columns:
        trend = df.groupby("record_year", as_index=False)[TARGET_COL].mean()
        trend = trend.sort_values("record_year")
        plt.figure(figsize=(8, 5))
        plt.plot(trend["record_year"], trend[TARGET_COL], marker="o")
        plt.title("Average Weekly HRI Rate by Year")
        plt.xlabel("Year")
        plt.ylabel("Mean Weekly HRI Rate")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "viz_target_trend_by_year.png", dpi=180)
        plt.close()

    # 3) Top feature-target correlations (bar)
    rank = (
        corr[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
        .head(8)
        .sort_values(ascending=True)
    )
    plt.figure(figsize=(9, 5))
    plt.barh(rank.index, rank.values)
    plt.title("Absolute Correlation with Target (Top Features)")
    plt.xlabel("|Correlation|")
    plt.tight_layout()
    plt.savefig(out_dir / "viz_top_feature_correlations.png", dpi=180)
    plt.close()

    # 4) Predicted vs actual on holdout split
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    mn = min(float(np.min(y_test)), float(np.min(y_pred)))
    mx = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title("Predicted vs Actual (Holdout Set)")
    plt.xlabel("Actual HRI Rate")
    plt.ylabel("Predicted HRI Rate")
    plt.tight_layout()
    plt.savefig(out_dir / "viz_predicted_vs_actual.png", dpi=180)
    plt.close()

    # 5) HRI rate distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET_COL], bins=20)
    plt.title("Distribution of Weekly HRI Rate")
    plt.xlabel("Weekly HRI Rate")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "viz_target_distribution.png", dpi=180)
    plt.close()

    print("Saved visualizations to:", out_dir)
    for p in sorted(out_dir.glob("viz_*.png")):
        print("-", p.name)


if __name__ == "__main__":
    main()
