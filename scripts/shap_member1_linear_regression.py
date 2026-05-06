"""
SHAP plots for the Member 1 linear regression model (LinearExplainer).

Uses synthetic training rows as the SHAP background distribution (same workflow as training).

Usage (from project root):
  python3 scripts/shap_member1_linear_regression.py

Defaults target the 4-feature LR model aligned with the frozen DT feature set.

Outputs (do not overwrite DT SHAP files):
  docs/analysis_outputs/shap_summary_bar_member1_linear.png
  docs/analysis_outputs/shap_force_member1_linear.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import pandas as pd
import shap

# Reuse the same force-plot helpers as the DT script (layout + labels).
from shap_member1_dt import (  # noqa: E402  # type: ignore
    _annotate_force_segment_values,
    _multiline_feature_labels,
    _style_base_and_output_labels,
    _thicken_force_bar_geometry,
    _tighten_force_x_limits,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="SHAP plots for Member 1 LinearRegression.")
    p.add_argument(
        "--model",
        default="experiments/linear_regression/outputs/linear_regression_4feat_same_as_frozen_dt.pkl",
        help="LinearRegression .pkl under repo root.",
    )
    p.add_argument(
        "--metrics",
        default="experiments/linear_regression/outputs/linear_regression_4feat_same_as_frozen_dt_metrics.json",
        help="Metrics JSON with feature_cols (must match the model).",
    )
    p.add_argument(
        "--synthetic",
        default="data/synthetic_hri_dataset_fixed.csv",
        help="Background data for LinearExplainer (train distribution).",
    )
    p.add_argument("--real", default="data/final_hri_modeling_dataset.csv")
    p.add_argument(
        "--background-rows",
        type=int,
        default=2000,
        help="Max synthetic rows for SHAP background (LinearExplainer reference data).",
    )
    p.add_argument("--sample", type=int, default=400, help="Max rows from real CSV for plots.")
    p.add_argument("--force-row", type=int, default=0, help="Row index for horizontal force plot.")
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

    synth_path = root / args.synthetic
    real_path = root / args.real
    train_df = pd.read_csv(synth_path)
    real = pd.read_csv(real_path)

    for name, df in [("synthetic", train_df), ("real", real)]:
        missing = [c for c in feature_cols + ["hri_value"] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    X_bg = train_df[feature_cols].copy()
    ok_bg = X_bg.notna().all(axis=1)
    X_bg = X_bg.loc[ok_bg]
    if len(X_bg) > args.background_rows:
        X_bg = X_bg.sample(n=args.background_rows, random_state=42)

    X = real[feature_cols].copy()
    if len(X) > args.sample:
        X = X.sample(n=args.sample, random_state=42)

    explainer = shap.LinearExplainer(model, X_bg)
    shap_values = explainer.shap_values(X)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = out_dir / "shap_summary_bar_member1_linear.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()

    row = int(args.force_row)
    if row < 0 or row >= len(X):
        raise ValueError(f"--force-row must be in [0, {len(X) - 1}], got {row}")

    ev = explainer.expected_value
    base = float(np.asarray(ev, dtype=float).reshape(-1)[0])
    phi_row = np.asarray(shap_values[row], dtype=float).reshape(-1)
    pred_val = float(base + phi_row.sum())

    shap.force_plot(
        base,
        phi_row,
        X.iloc[row],
        matplotlib=True,
        show=False,
        figsize=(15.5, 5.6),
        out_names=[r"$f(x)$"],
        contribution_threshold=1e-12,
    )
    fig = plt.gcf()
    ax = plt.gca()

    _tiny_key = {"higher", "lower", r"$\leftarrow$", r"$\rightarrow$"}
    for t in list(ax.texts):
        if t.get_text().strip() in _tiny_key:
            t.remove()

    _multiline_feature_labels(ax)
    _thicken_force_bar_geometry(ax)
    _tighten_force_x_limits(ax)
    _style_base_and_output_labels(ax, base, pred_val)
    _annotate_force_segment_values(ax, phi_row, pred_val)

    fig.subplots_adjust(left=0.06, right=0.96, top=0.56, bottom=0.16)

    fig.suptitle(
        "FORCE PLOT (Linear Regression) — Single Prediction Breakdown",
        x=0.05,
        y=0.96,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#111111",
    )

    y_arrow = 0.73
    y_caption = 0.67

    def _banner_arrow(tail_xy, head_xy, color):
        p = FancyArrowPatch(
            tail_xy,
            head_xy,
            transform=fig.transFigure,
            arrowstyle="-|>",
            linewidth=4.2,
            mutation_scale=21,
            mutation_aspect=1.1,
            color=color,
            clip_on=False,
            zorder=100,
        )
        fig.add_artist(p)

    _banner_arrow((0.30, y_arrow), (0.07, y_arrow), "#d32f2f")
    fig.text(
        0.072,
        y_caption,
        "Pushes prediction lower",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#c62828",
    )

    _banner_arrow((0.70, y_arrow), (0.93, y_arrow), "#1976d2")
    fig.text(
        0.928,
        y_caption,
        "Pushes prediction higher",
        ha="right",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#1565c0",
    )

    force_path = out_dir / "shap_force_member1_linear.png"
    ax.set_ylim(-0.38, 0.22)
    plt.savefig(force_path, dpi=220, bbox_inches="tight", pad_inches=0.34)
    plt.close()

    print("Wrote:", bar_path.relative_to(root))
    print("Wrote:", force_path.relative_to(root))


if __name__ == "__main__":
    main()
