"""
Export several matplotlib images of the frozen Member 1 decision tree.

Reads feature names from models/member1_decision_tree_metrics.json.

Usage (from repo root):
  python3 scripts/visualize_member1_decision_tree.py

Writes PNGs under docs/analysis_outputs/ and a text dump alongside them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.tree import export_text, plot_tree  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Plot frozen Member 1 decision tree.")
    p.add_argument("--model", default="models/member1_decision_tree.pkl")
    p.add_argument("--metrics", default="models/member1_decision_tree_metrics.json")
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

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "member1_decision_tree_viz"

    def _plot(
        name: str,
        *,
        max_depth: int | None,
        figsize: tuple[float, float],
        fontsize: int,
        dpi: int,
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        plot_tree(
            model,
            feature_names=feature_cols,
            filled=True,
            rounded=True,
            impurity=True,
            precision=2,
            fontsize=fontsize,
            max_depth=max_depth,
            ax=ax,
        )
        ax.set_title(
            f"Member 1 DecisionTreeRegressor (frozen) — display depth cap: {max_depth if max_depth is not None else 'full'}",
            fontsize=12,
        )
        fig.tight_layout()
        path = out_dir / f"{base}_{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Wrote:", path.relative_to(root))

    # Versions: full + progressively truncated (easier to read in slides).
    _plot("full", max_depth=None, figsize=(34, 22), fontsize=8, dpi=160)
    _plot("depth_4", max_depth=4, figsize=(28, 18), fontsize=9, dpi=170)
    _plot("depth_3", max_depth=3, figsize=(22, 14), fontsize=10, dpi=180)
    _plot("depth_2_slide", max_depth=2, figsize=(16, 10), fontsize=11, dpi=200)

    # Compact full tree (smaller canvas; labels tighter — good for appendix).
    fig, ax = plt.subplots(figsize=(26, 18), facecolor="white")
    plot_tree(
        model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        impurity=False,
        precision=1,
        fontsize=7,
        max_depth=None,
        ax=ax,
    )
    ax.set_title("Member 1 DecisionTreeRegressor — full tree (no impurity boxes)", fontsize=11)
    fig.tight_layout()
    compact_path = out_dir / f"{base}_full_no_impurity.png"
    fig.savefig(compact_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote:", compact_path.relative_to(root))

    text_dump = export_text(
        model,
        feature_names=list(feature_cols),
        decimals=2,
        show_weights=True,
    )
    txt_path = out_dir / f"{base}_structure.txt"
    txt_path.write_text(text_dump, encoding="utf-8")
    print("Wrote:", txt_path.relative_to(root))


if __name__ == "__main__":
    main()
