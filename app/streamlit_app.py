from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Flat zip layout: all paths are filenames next to this script.
SYNTHETIC_CSV = "synthetic_hri_dataset_fixed.csv"
REAL_CSV = "final_hri_modeling_dataset.csv"
MODEL_PATH = "mary_best_engineered_linear_model.pkl"
METRICS_PATH = "mary_best_engineered_linear_metrics.json"
MARY_SUBMITTED_4FEAT_JSON = "mary_submitted_linear_regression_4feat_metrics.json"
MARY_SUBMITTED_8FEAT_JSON = "mary_submitted_linear_regression_8feat_metrics.json"
MARY_USER_INPUT_COLS = ["min_temp_celsius", "feat_median_hh_income", "unsheltered_homeless"]


def project_root() -> Path:
    return Path(__file__).resolve().parent


def mary_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    mt = df["min_temp_celsius"].astype(float)
    inc = df["feat_median_hh_income"].astype(float)
    ush = df["unsheltered_homeless"].astype(float).clip(lower=0)
    return pd.DataFrame(
        {
            "min_temp_celsius": mt,
            "min_temp_sq": mt**2,
            "feat_median_hh_income": inc,
            "log_unsheltered_homeless": np.log1p(ush),
            "temp_income_interaction": mt * inc,
        }
    )


@st.cache_resource
def load_model():
    return joblib.load(project_root() / MODEL_PATH)


_INPUT_DEFAULTS = {
    "min_temp_celsius": 15.0,
    "feat_median_hh_income": 70000.0,
    "unsheltered_homeless": 15000.0,
}


def metric_card(metrics_path: Path) -> None:
    if not metrics_path.exists():
        st.info("No metrics JSON next to the model.")
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics.get('mae', 0):.2f}")
    c2.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
    c3.metric("R²", f"{metrics.get('r2', 0):.3f}")


def apply_custom_style(theme_mode: str) -> None:
    accent = "#82AAFF"
    subtle = "#B8C1EC"
    panel_bg = "#1F2430"
    body_glow = "rgba(130, 170, 255, 0.06)"
    tab_underline = "#82AAFF"

    if theme_mode == "Black + Yellow":
        accent = "#FFD60A"
        subtle = "#FFF1A8"
        panel_bg = "#000000"
        body_glow = "rgba(255, 214, 10, 0.09)"
        tab_underline = "#FFD60A"
    elif theme_mode == "Neon (Blue + Purple)":
        accent = "#7C3AED"
        subtle = "#C4B5FD"
        panel_bg = "#18122B"
        body_glow = "rgba(124, 58, 237, 0.09)"
        tab_underline = "#7C3AED"

    css = """
        <style>
            .block-container {padding-top: 1.25rem; padding-bottom: 1.5rem;}
            section[data-testid="stSidebar"] {
                background: __PANEL_BG__ !important;
                border-right: 1px solid __ACCENT__66 !important;
            }
            .custom-subtle {
                color: __SUBTLE__ !important;
                font-size: 0.95rem;
                margin-top: -0.3rem;
                margin-bottom: 0.8rem;
            }
            div[data-testid="stMetric"] {
                border: 1px solid __ACCENT__55 !important;
                border-radius: 12px;
                padding: 0.5rem 0.8rem;
                background: __BODY_GLOW__ !important;
            }
            h1, h2, h3 { color: __SUBTLE__ !important; }
            button[kind="secondary"] { border-color: __ACCENT__ !important; }
            button[kind="secondary"]:hover {
                color: black !important;
                background: __ACCENT__ !important;
            }
            div[data-baseweb="tab-highlight"] {
                background-color: __TAB_UNDERLINE__ !important;
            }
            div[data-baseweb="tab"] > div {
                color: __SUBTLE__ !important;
            }
            div[data-baseweb="select"] * {
                border-color: __ACCENT__88 !important;
            }
            *[style*="#ff4b4b"], *[style*="rgb(255, 75, 75)"] {
                color: __ACCENT__ !important;
                border-color: __ACCENT__ !important;
            }
        </style>
    """
    css = (
        css.replace("__SUBTLE__", subtle)
        .replace("__ACCENT__", accent)
        .replace("__PANEL_BG__", panel_bg)
        .replace("__BODY_GLOW__", body_glow)
        .replace("__TAB_UNDERLINE__", tab_underline)
    )
    st.markdown(css, unsafe_allow_html=True)


def sidebar_controls() -> dict:
    st.sidebar.header("Display Controls")
    theme_mode = st.sidebar.selectbox(
        "Theme Accent",
        ["Default Blue", "Black + Yellow", "Neon (Blue + Purple)"],
        index=0,
    )
    show_dataset_preview = st.sidebar.checkbox("Show dataset preview", value=True)
    max_batch_rows = st.sidebar.slider("Batch preview rows", min_value=5, max_value=100, value=20, step=5)
    return {
        "theme_mode": theme_mode,
        "show_dataset_preview": show_dataset_preview,
        "max_batch_rows": max_batch_rows,
    }


def manual_prediction(model) -> None:
    st.subheader("Manual Input Prediction")
    st.markdown(
        "<div class='custom-subtle'>Three inputs; app builds engineered features (squares, log1p, interaction) then predicts.</div>",
        unsafe_allow_html=True,
    )
    with st.form("manual_pred_form"):
        left, right, _ = st.columns(3)
        with left:
            mt = st.number_input(
                "min_temp_celsius",
                value=float(_INPUT_DEFAULTS["min_temp_celsius"]),
                key="m_mt",
            )
        with right:
            inc = st.number_input(
                "feat_median_hh_income",
                min_value=1.0,
                value=float(_INPUT_DEFAULTS["feat_median_hh_income"]),
                key="m_inc",
            )
        ush = st.number_input(
            "unsheltered_homeless",
            min_value=0.0,
            value=float(_INPUT_DEFAULTS["unsheltered_homeless"]),
            key="m_ush",
        )
        submitted = st.form_submit_button("Predict HRI value")
    if submitted:
        raw = pd.DataFrame([[mt, inc, ush]], columns=MARY_USER_INPUT_COLS)
        X = mary_engineered_features(raw)
        pred = float(model.predict(X)[0])
        st.success(f"Predicted HRI value: {pred:.2f}")


def csv_prediction(model, max_rows: int) -> None:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "<div class='custom-subtle'>CSV must include: min_temp_celsius, feat_median_hh_income, unsheltered_homeless.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if uploaded is None:
        return
    df = pd.read_csv(uploaded)
    missing = [c for c in MARY_USER_INPUT_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    X = mary_engineered_features(df[MARY_USER_INPUT_COLS])
    preds = model.predict(X)
    out = df.copy()
    out["predicted_hri_value"] = preds
    st.dataframe(out.head(max_rows), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions_hri.csv",
        mime="text/csv",
    )


def visuals_tab_regression_only(root: Path, metrics_path: Path) -> None:
    st.subheader("Linear regression visuals")
    st.markdown(
        "<div class='custom-subtle'>Intercept and coefficients for the deployed engineered linear model.</div>",
        unsafe_allow_html=True,
    )

    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.metric("Intercept", f"{m.get('intercept', 0):.6f}")
        coefs = m.get("coefficients") or {}
        if coefs:
            coef_df = pd.DataFrame([coefs]).T.rename(columns={0: "coefficient"})
            st.dataframe(coef_df, use_container_width=True)
            chart_df = coef_df.copy()
            chart_df["abs"] = chart_df["coefficient"].abs()
            st.markdown("**Coefficient magnitude (absolute value)**")
            st.bar_chart(chart_df["abs"])

    with st.expander("Mary's OLS runs (4-feature and 8-feature metrics from JSON)"):
        p4 = root / MARY_SUBMITTED_4FEAT_JSON
        p8 = root / MARY_SUBMITTED_8FEAT_JSON
        if p4.exists():
            j = json.loads(p4.read_text(encoding="utf-8"))
            tr = j.get("test_on_real") or {}
            st.markdown("**4-feature linear regression (real holdout)**")
            st.write(
                f"MAE {tr.get('mae', j.get('mae')):.4f}, "
                f"RMSE {tr.get('rmse', j.get('rmse')):.4f}, "
                f"R² {tr.get('r2', j.get('r2')):.4f}"
            )
            c4 = j.get("coefficients") or {}
            if c4:
                st.dataframe(pd.DataFrame([c4]).T.rename(columns={0: "coefficient"}))
        if p8.exists():
            j = json.loads(p8.read_text(encoding="utf-8"))
            tr = j.get("test_on_real") or {}
            st.markdown("**8-feature linear regression (real holdout)**")
            st.write(
                f"MAE {tr.get('mae', j.get('mae')):.4f}, "
                f"RMSE {tr.get('rmse', j.get('rmse')):.4f}, "
                f"R² {tr.get('r2', j.get('r2')):.4f}"
            )
            c8 = j.get("coefficients") or {}
            if c8:
                st.dataframe(pd.DataFrame([c8]).T.rename(columns={0: "coefficient"}))


def main() -> None:
    st.set_page_config(
        page_title="HRI predictor — linear regression",
        layout="wide",
    )
    root = project_root()
    ui = sidebar_controls()
    apply_custom_style(ui["theme_mode"])

    st.title("Weekly HRI prediction (heat-affected regions)")
    st.caption(
        "Deployed model: Mary's engineered linear regression. Target: `hri_value`. "
        "See `dataset_schema.txt`."
    )

    synth_path = root / SYNTHETIC_CSV
    real_path = root / REAL_CSV
    model_path = root / MODEL_PATH
    metrics_path = root / METRICS_PATH

    tab_overview, tab_manual, tab_batch, tab_visuals = st.tabs(
        ["Overview", "Manual Prediction", "Batch Prediction", "Visuals"]
    )

    with tab_overview:
        st.subheader("Data + status")
        c1, c2 = st.columns(2)
        synth_rows = len(pd.read_csv(synth_path)) if synth_path.exists() else None
        real_rows = len(pd.read_csv(real_path)) if real_path.exists() else None
        c1.metric("Synthetic train rows", str(synth_rows) if synth_rows is not None else "—")
        c2.metric("Real evaluation rows", str(real_rows) if real_rows is not None else "—")

        metric_card(metrics_path)

        if not model_path.exists():
            st.warning(f"Missing `{MODEL_PATH}` in this folder.")
        else:
            st.success("Model file found — prediction tabs are enabled.")

        if ui["show_dataset_preview"]:
            if synth_path.exists():
                st.markdown("**Synthetic preview**")
                st.dataframe(pd.read_csv(synth_path).head(8), use_container_width=True)
            if real_path.exists():
                st.markdown("**Real evaluation preview**")
                st.dataframe(pd.read_csv(real_path).head(8), use_container_width=True)

    if not model_path.exists():
        with tab_manual:
            st.info("Add `mary_best_engineered_linear_model.pkl` next to `streamlit_app.py`.")
        with tab_batch:
            st.info("Add `mary_best_engineered_linear_model.pkl` next to `streamlit_app.py`.")
        return

    model = load_model()
    with tab_manual:
        manual_prediction(model)
    with tab_batch:
        csv_prediction(model, ui["max_batch_rows"])
    with tab_visuals:
        visuals_tab_regression_only(root, metrics_path)


if __name__ == "__main__":
    main()
