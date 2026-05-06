"""
Streamlit app: Mary's engineered Linear Regression (best RMSE in team comparison).

User enters raw inputs:
  min_temp_celsius, feat_median_hh_income, unsheltered_homeless

The app builds engineered columns exactly as training:
  min_temp_sq, log_unsheltered_homeless, temp_income_interaction

Run locally (repo root):
  streamlit run app/streamlit_mary_lr_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_REL = "models/mary_best_engineered_linear_regression.joblib"
METRICS_REL = "models/mary_best_engineered_linear_regression_metrics.json"


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def engineer_row(
    min_temp: float, income: float, unsheltered: float
) -> pd.DataFrame:
    min_temp_sq = min_temp**2
    log_uh = float(np.log1p(max(unsheltered, 0.0)))
    interaction = min_temp * income
    cols = [
        "min_temp_celsius",
        "min_temp_sq",
        "feat_median_hh_income",
        "log_unsheltered_homeless",
        "temp_income_interaction",
    ]
    data = [[min_temp, min_temp_sq, income, log_uh, interaction]]
    return pd.DataFrame(data, columns=cols)


def risk_band(hri_pred: float) -> str:
    if hri_pred <= 29:
        return "Low"
    if hri_pred <= 166:
        return "Moderate"
    if hri_pred <= 363:
        return "High"
    return "Very high"


@st.cache_resource
def load_estimator(path: Path):
    return joblib.load(path)


def main() -> None:
    st.set_page_config(
        page_title="HRI — Engineered Linear Regression",
        layout="wide",
    )
    root = root_dir()
    model_path = root / MODEL_REL
    metrics_path = root / METRICS_REL

    st.title("Weekly HRI prediction — engineered Linear Regression")
    st.caption(
        "Trained on synthetic data; metrics reported on locked real holdout. "
        "Enter three raw features — engineered terms are computed automatically."
    )
    st.info(
        "Predicted HRI interpretation bands: "
        "Low (0-29), Moderate (30-166), High (167-363), Very high (>363)."
    )

    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout MAE", f"{m.get('mae', 0):.2f}")
        c2.metric("Holdout RMSE", f"{m.get('rmse', 0):.2f}")
        c3.metric("Holdout R²", f"{m.get('r2', 0):.3f}")

    if not model_path.exists():
        st.error(
            f"Missing model file `{MODEL_REL}`. Run:\n\n"
            "`python scripts/train_and_save_mary_best_lr.py`\n\n"
            "from the project root, then refresh this app."
        )
        return

    model = load_estimator(model_path)

    tab_manual, tab_batch = st.tabs(["Manual prediction", "Batch CSV"])

    with tab_manual:
        st.subheader("Inputs")
        with st.form("predict"):
            min_temp = st.number_input("min_temp_celsius", value=-2.1)
            income = st.number_input("feat_median_hh_income", value=65780.75)
            unsheltered = st.number_input(
                "unsheltered_homeless",
                min_value=0.0,
                value=26140.0,
                help="Example default matches Mary's sample_input_output.csv sanity row.",
            )
            submitted = st.form_submit_button("Predict HRI")
        if submitted:
            X = engineer_row(min_temp, income, unsheltered)
            pred = float(model.predict(X)[0])
            st.success(f"Predicted **hri_value**: **{pred:.4f}**")
            st.markdown(f"**Predicted risk band:** `{risk_band(pred)}`")

    with tab_batch:
        st.markdown(
            "Upload CSV with columns **`min_temp_celsius`**, **`feat_median_hh_income`**, **`unsheltered_homeless`**."
        )
        up = st.file_uploader("CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            need = ["min_temp_celsius", "feat_median_hh_income", "unsheltered_homeless"]
            miss = [c for c in need if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                eng = pd.DataFrame(
                    {
                        "min_temp_celsius": df["min_temp_celsius"].astype(float),
                        "min_temp_sq": df["min_temp_celsius"].astype(float) ** 2,
                        "feat_median_hh_income": df["feat_median_hh_income"].astype(
                            float
                        ),
                        "log_unsheltered_homeless": np.log1p(
                            df["unsheltered_homeless"].clip(lower=0).astype(float)
                        ),
                        "temp_income_interaction": df["min_temp_celsius"].astype(float)
                        * df["feat_median_hh_income"].astype(float),
                    }
                )
                df["predicted_hri_value"] = model.predict(eng)
                st.dataframe(df.head(50), use_container_width=True)
                st.download_button(
                    "Download predictions",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="mary_lr_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
