from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

FEATURE_COLS = [
    "max_temp_celsius",
    "min_temp_celsius",
    "feat_poverty_rate",
    "feat_unemployment_rate",
    "feat_median_hh_income",
    "feat_total_population",
    "overall_homeless",
    "unsheltered_homeless",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read feature input CSV, load trained model, and write predictions CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing the 8 feature columns used in training.",
    )
    parser.add_argument(
        "--model",
        default="models/member1_decision_tree.pkl",
        help="Path to trained model .pkl file.",
    )
    parser.add_argument(
        "--output",
        default="models/predictions_from_file.csv",
        help="Path to output predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df = pd.read_csv(input_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    model = joblib.load(model_path)
    preds = model.predict(df[FEATURE_COLS])

    out = df.copy()
    out["predicted_hri_value"] = preds
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Predictions written to: {output_path}")
    print(f"Rows predicted: {len(out)}")


if __name__ == "__main__":
    main()
