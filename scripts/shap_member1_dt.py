"""
SHAP plots for the saved Member 1 Decision Tree (TreeExplainer).

Reads feature column names from models/member1_decision_tree_metrics.json so they
stay in sync after feature-drop experiments.

Usage (from project root):
  pip install shap
  python3 scripts/shap_member1_dt.py

Outputs:
  docs/analysis_outputs/shap_summary_bar_member1.png
  docs/analysis_outputs/shap_force_member1.png  (horizontal SHAP force plot for one row)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import pandas as pd
import shap


def _fmt_effect(v: float) -> str:
    """Compact +/- labels inside force segments (HRI-scale friendly)."""
    av = abs(float(v))
    if av >= 100:
        return f"{float(v):+.0f}"
    if av >= 10:
        return f"{float(v):+.1f}"
    return f"{float(v):+.2f}"


def _annotate_force_segment_values(
    ax,
    phi: np.ndarray,
    pred: float,
) -> None:
    """SHAP's matplotlib force renderer draws chevrons but not contribution amounts inside segments."""
    phi = np.asarray(phi, dtype=float).ravel()
    pred = float(pred)
    span_axis = abs(ax.get_xlim()[1] - ax.get_xlim()[0])

    def _place(mx: float, eff: float, seg_width: float) -> None:
        if seg_width <= 1e-9 * max(abs(pred), 1.0):
            return
        fs = min(16.0, max(12.0, (seg_width / max(span_axis, 1e-9)) * 1020.0))
        ax.text(
            mx,
            0.057,
            _fmt_effect(eff),
            fontsize=fs,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=25,
            clip_on=False,
        )

    neg_ix = np.where(phi < 0)[0]
    neg_ix = neg_ix[np.argsort(phi[neg_ix])]
    x_right = pred
    for i in neg_ix:
        eff = float(phi[i])
        x_left = x_right + eff
        _place((x_left + x_right) / 2.0, eff, abs(x_right - x_left))
        x_right = x_left

    pos_ix = np.where(phi >= 0)[0]
    pos_ix = pos_ix[np.argsort(-phi[pos_ix])]
    x_left = pred
    for i in pos_ix:
        eff = float(phi[i])
        x_right = x_left + eff
        _place((x_left + x_right) / 2.0, eff, abs(x_right - x_left))
        x_left = x_right


def _multiline_feature_labels(ax) -> None:
    """Course-style mock: feature name on first line, instance value on second."""
    for t in list(ax.texts):
        s = t.get_text()
        if s.startswith("$") or "=" not in s:
            continue
        parts = [p.strip() for p in s.split("=", 1)]
        if len(parts) != 2 or not parts[0]:
            continue
        name, val = parts
        t.set_text(f"{name}\n{val}")
        t.set_fontsize(15)
        t.set_linespacing(1.1)


def _thicken_force_bar_geometry(ax, scale: float = 1.55) -> None:
    """Increase vertical thickness of chevrons/shading to reduce dead space."""
    for patch in list(ax.patches):
        get_xy = getattr(patch, "get_xy", None)
        set_xy = getattr(patch, "set_xy", None)
        if get_xy is None or set_xy is None:
            continue
        try:
            verts = np.array(get_xy(), dtype=float)
        except Exception:
            continue
        if verts.ndim != 2 or verts.shape[1] != 2:
            continue
        # Expand around the original bar midline (default bars run from y=0 to y=0.1).
        y_mid = 0.05
        verts[:, 1] = y_mid + (verts[:, 1] - y_mid) * scale
        try:
            set_xy(verts)
        except Exception:
            continue


def _tighten_force_x_limits(ax, pad_frac: float = 0.04) -> None:
    """Trim empty horizontal space by fitting axis to actual force-bar geometry."""
    xs: list[float] = []
    for patch in list(ax.patches):
        get_xy = getattr(patch, "get_xy", None)
        if get_xy is None:
            continue
        try:
            verts = np.array(get_xy(), dtype=float)
        except Exception:
            continue
        if verts.ndim != 2 or verts.shape[1] != 2:
            continue
        xs.extend(verts[:, 0].tolist())
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    span = max(1e-9, xmax - xmin)
    pad = span * pad_frac
    ax.set_xlim(xmin - pad, xmax + pad)


def _style_base_and_output_labels(ax, base_val: float, pred_val: float) -> None:
    """Course mock wording: $E[f(x)]$ + numeric base; $f(x)$ line above bold prediction."""
    base_val = float(base_val)
    pred_val = float(pred_val)
    tol = max(0.05, abs(pred_val) * 1e-5)

    for t in list(ax.texts):
        if t.get_text() == "base value":
            t.set_text(r"$E[f(x)]$")
            t.set_fontsize(11)
            t.set_alpha(1.0)
            t.set_position((base_val, 0.31))
            ax.text(
                base_val,
                0.215,
                f"{base_val:.2f}",
                fontsize=13,
                ha="center",
                va="center",
                fontweight="bold",
                color="#222222",
                clip_on=False,
            )
            continue

        txt = t.get_text()
        if "$f(x)$" in txt:
            pos = t.get_position()
            t.set_fontsize(11)
            t.set_alpha(0.55)
            t.set_position((pos[0], 0.345))
            continue

        try:
            v = float(txt)
        except ValueError:
            continue

        _, y = t.get_position()
        # SHAP draws the predicted value near ~0.25; ignore axis tick labels farther from that band.
        if abs(v - pred_val) <= tol and 0.18 <= y <= 0.32:
            t.set_fontsize(15)
            t.set_fontweight("bold")
            pos = t.get_position()
            t.set_position((pos[0], 0.248))


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
    p.add_argument(
        "--force-row",
        type=int,
        default=0,
        help="Row index (within sampled X) for horizontal force_plot.",
    )
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

    # Horizontal force layout (matplotlib) — classic SHAP force_plot, not waterfall.
    row = int(args.force_row)
    if row < 0 or row >= len(X):
        raise ValueError(f"--force-row must be in [0, {len(X) - 1}], got {row}")
    base = float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0])
    phi_row = np.asarray(shap_values[row], dtype=float).reshape(-1)
    pred_val = float(base + phi_row.sum())

    # force_plot creates its own figure — do not plt.figure() first or decorations land on the wrong canvas.
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

    # Remove SHAP's tiny built-in "higher / lower" key; course mock uses banner arrows + phrases instead.
    _tiny_key = {"higher", "lower", r"$\leftarrow$", r"$\rightarrow$"}
    for t in list(ax.texts):
        if t.get_text().strip() in _tiny_key:
            t.remove()

    _multiline_feature_labels(ax)
    _thicken_force_bar_geometry(ax)
    _tighten_force_x_limits(ax)
    _style_base_and_output_labels(ax, base, pred_val)
    _annotate_force_segment_values(ax, phi_row, pred_val)

    # Pull chart upward and tighten the gap between banner and force bars.
    fig.subplots_adjust(left=0.06, right=0.96, top=0.56, bottom=0.16)

    fig.suptitle(
        "FORCE PLOT - Single Prediction Breakdown",
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

    force_path = out_dir / "shap_force_member1.png"
    ax.set_ylim(-0.38, 0.22)
    plt.savefig(force_path, dpi=220, bbox_inches="tight", pad_inches=0.34)
    plt.close()

    print("Wrote:", bar_path.relative_to(root))
    print("Wrote:", force_path.relative_to(root))


if __name__ == "__main__":
    main()
