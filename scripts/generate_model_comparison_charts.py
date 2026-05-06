from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Known values from current project artifacts + user-provided XGBoost tuned metrics.
    df = pd.DataFrame(
        [
            {
                "model": "Decision Tree",
                "baseline_rmse": 167.886778,
                "tuned_rmse": 152.936171,
                "tuned_mae": 91.790709,
                "tuned_r2": 0.351977,
                "interpretability": 9,
                "flexibility": 5,
            },
            {
                "model": "Linear Regression",
                "baseline_rmse": 174.908912,
                "tuned_rmse": 149.164207,
                "tuned_mae": 98.948976,
                "tuned_r2": 0.383548,
                "interpretability": 8,
                "flexibility": 4,
            },
            {
                "model": "XGBoost",
                "baseline_rmse": np.nan,  # Pending from teammate baseline run.
                "tuned_rmse": 155.16,
                "tuned_mae": 98.10,
                "tuned_r2": 0.333,
                "interpretability": 5,
                "flexibility": 9,
            },
        ]
    )

    # 1) Before vs after tuning dumbbell chart.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        b = row["baseline_rmse"]
        t = row["tuned_rmse"]
        if np.isfinite(b):
            ax.plot([b, t], [i, i], color="#777777", linewidth=2)
            ax.scatter(b, i, color="#d95f02", s=90, label="Baseline" if i == 0 else "")
        ax.scatter(t, i, color="#1b9e77", s=90, label="Tuned" if i == 0 else "")
        if not np.isfinite(b):
            ax.text(t + 0.8, i + 0.08, "baseline TBD", fontsize=9, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Before vs After Tuning (Dumbbell)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_dumbbell_baseline_vs_tuned_rmse.png", dpi=220)
    plt.close(fig)

    # 2) Tuned RMSE bar chart.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = df.sort_values("tuned_rmse")
    bars = ax.bar(order["model"], order["tuned_rmse"], color=["#4daf4a", "#377eb8", "#ff7f00"])
    ax.set_ylabel("Tuned RMSE (lower is better)")
    ax.set_title("Tuned RMSE by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, order["tuned_rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_bar_tuned_rmse_ranking.png", dpi=220)
    plt.close(fig)

    # 3) Error reduction % chart.
    valid = df[np.isfinite(df["baseline_rmse"])].copy()
    valid["error_reduction_pct"] = (valid["baseline_rmse"] - valid["tuned_rmse"]) / valid["baseline_rmse"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(valid["model"], valid["error_reduction_pct"], color="#5e3c99")
    ax.set_ylabel("Error Reduction %")
    ax.set_title("RMSE Reduction After Tuning")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, valid["error_reduction_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_bar_error_reduction_percent.png", dpi=220)
    plt.close(fig)

    # 4) Tradeoff bubble chart: x=RMSE, y=R2, bubble size=MAE.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sizes = df["tuned_mae"] * 8
    ax.scatter(df["tuned_rmse"], df["tuned_r2"], s=sizes, alpha=0.55, c=["#66c2a5", "#fc8d62", "#8da0cb"])
    for _, row in df.iterrows():
        ax.text(row["tuned_rmse"] + 0.25, row["tuned_r2"] + 0.001, row["model"], fontsize=9)
    ax.set_xlabel("Tuned RMSE (lower better)")
    ax.set_ylabel("Tuned R² (higher better)")
    ax.set_title("Performance Tradeoff (Bubble Size = MAE)")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "chart_bubble_rmse_r2_mae_tradeoff.png", dpi=220)
    plt.close(fig)

    # 5) Model complexity panel: interpretability vs flexibility scale.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ranked = df.sort_values("interpretability", ascending=True)
    axes[0].barh(ranked["model"], ranked["interpretability"], color="#1f78b4")
    axes[0].set_title("Interpretability (higher = easier to explain)")
    axes[0].set_xlim(0, 10)
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    ranked2 = df.sort_values("flexibility", ascending=True)
    axes[1].barh(ranked2["model"], ranked2["flexibility"], color="#33a02c")
    axes[1].set_title("Flexibility (higher = more nonlinear power)")
    axes[1].set_xlim(0, 10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Model Complexity Panel")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_model_complexity_panel.png", dpi=220)
    plt.close(fig)

    print("Saved chart pack to:", out_dir)


if __name__ == "__main__":
    main()
