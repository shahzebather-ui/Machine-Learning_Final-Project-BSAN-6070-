from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REAL = "data/final_hri_modeling_dataset.csv"
DEFAULT_SYNTH = "data/synthetic_hri_dataset_fixed.csv"
TARGET_COL = "hri_value"

META_EXCLUDE = {
    TARGET_COL,
    "regionid",
    "region_label",
    "record_year",
    "week",
    "state",
    "city",
    "is_synthetic",
    "source",
    "synthetic",
}


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted(c for c in numeric if c not in META_EXCLUDE)


def summarize_series(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare distributions of real vs synthetic CSVs (alignment check)."
    )
    parser.add_argument("--real", default=DEFAULT_REAL, help="Path to real/holdout CSV")
    parser.add_argument("--synthetic", default=DEFAULT_SYNTH, help="Path to synthetic train CSV")
    parser.add_argument(
        "--out",
        default="models/synthetic_vs_real_summary.json",
        help="Where to write JSON summary",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    real_path = root / args.real
    synth_path = root / args.synthetic
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(real_path)
    synth = pd.read_csv(synth_path)

    real_feats = set(infer_feature_cols(real))
    synth_feats = set(infer_feature_cols(synth))
    common = sorted(real_feats & synth_feats)

    missing_in_synth = sorted(real_feats - synth_feats)
    missing_in_real = sorted(synth_feats - real_feats)

    report: dict = {
        "real_path": str(real_path),
        "synthetic_path": str(synth_path),
        "n_real": int(len(real)),
        "n_synthetic": int(len(synth)),
        "common_numeric_features": common,
        "only_in_real": missing_in_synth,
        "only_in_synthetic": missing_in_real,
        "target": {},
        "features": {},
    }

    if TARGET_COL in real.columns and TARGET_COL in synth.columns:
        report["target"] = {
            "real": summarize_series(real[TARGET_COL]),
            "synthetic": summarize_series(synth[TARGET_COL]),
        }

    for col in common:
        report["features"][col] = {
            "real": summarize_series(real[col]),
            "synthetic": summarize_series(synth[col]),
        }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
