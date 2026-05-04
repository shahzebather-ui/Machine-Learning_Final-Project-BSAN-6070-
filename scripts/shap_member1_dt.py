"""
SHAP plots for the saved Member 1 Decision Tree (TreeExplainer).

Reads feature column names from models/member1_decision_tree_metrics.json so they
stay in sync after feature-drop experiments.

Usage (from project root):
  pip install shap
  python3 scripts/shap_member1_dt.py

Outputs:
  docs/analysis_outputs/shap_summary_bar_member1.png
  docs/analysis_outputs/shap_summary_beeswarm_member1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import shap


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="SHAP summary plots for Member 1 DT.")
    p.add_argument("--model", default="models/member1_decision_tree.pkl")
    p.add_argument("--real", default="data/final_hri_modeling_dataset.csv")
    p.add_argument(
        "--metrics",
        default="models/member1_decision_tree_metrics.json",
        help="JSON with feature_cols list (must match trained model).",
    )
    p.add_argument("--sample", type=int, default=400, help="Max rows from real CSV.")
    p.add_argument("--out-dir", default="docs/analysis_outputs")
    args = p.parse_args()

    model_path = root / args.model
    metrics_path = root / args.metrics
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols")
    if not feature_cols:
        raise ValueError("metrics JSON missing feature_cols")

    model = joblib.load(model_path)
    real = pd.read_csv(root / args.real)
    missing = [c for c in feature_cols if c not in real.columns]
    if missing:
        raise ValueError(f"Real CSV missing columns: {missing}")

    X = real[feature_cols].copy()
    if len(X) > args.sample:
        X = X.sample(n=args.sample, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = out_dir / "shap_summary_bar_member1.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    bee_path = out_dir / "shap_summary_beeswarm_member1.png"
    plt.savefig(bee_path, dpi=200)
    plt.close()

    print("Wrote:", bar_path.relative_to(root))
    print("Wrote:", bee_path.relative_to(root))


if __name__ == "__main__":
    main()
