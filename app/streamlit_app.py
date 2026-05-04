from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


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

SYNTHETIC_CSV = "data/synthetic_hri_dataset_fixed.csv"
REAL_CSV = "data/final_hri_modeling_dataset.csv"
MODEL_PATH = "models/member1_decision_tree.pkl"
METRICS_PATH = "models/member1_decision_tree_metrics.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def feature_columns(root: Path) -> list[str]:
    """Match training: use `feature_cols` from metrics JSON if present (after feature-drop)."""
    mp = root / METRICS_PATH
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        cols = m.get("feature_cols")
        if isinstance(cols, list) and cols:
            return cols
    return list(FEATURE_COLS)


# Defaults for manual form (only these labels get special ranges; others use generic defaults)
_INPUT_DEFAULTS = {
    "max_temp_celsius": 30.0,
    "min_temp_celsius": 15.0,
    "feat_poverty_rate": 0.14,
    "feat_unemployment_rate": 0.06,
    "feat_median_hh_income": 70000.0,
    "feat_total_population": 5.0e7,
    "overall_homeless": 50000.0,
    "unsheltered_homeless": 15000.0,
    "regionid": 4.0,
}


def metric_card(metrics_path: Path) -> None:
    if not metrics_path.exists():
        st.info("No metrics JSON yet. Train on synthetic data and save metrics next to the model.")
        return
    metrics = json.loads(metrics_path.read_text())
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


def manual_prediction(model, feature_cols: list[str]) -> None:
    st.subheader("Manual Input Prediction")
    st.markdown(
        "<div class='custom-subtle'>Enter feature values and run one prediction.</div>",
        unsafe_allow_html=True,
    )
    with st.form("manual_pred_form"):
        values = {}
        left, right = st.columns(2)
        mid = (len(feature_cols) + 1) // 2
        for i, col in enumerate(feature_cols):
            box = left if i < mid else right
            with box:
                base = float(_INPUT_DEFAULTS.get(col, 0.0))
                if col in ("feat_poverty_rate", "feat_unemployment_rate"):
                    values[col] = st.number_input(
                        col, min_value=0.0, max_value=1.0, value=base, key=f"in_{col}"
                    )
                elif col == "feat_total_population":
                    values[col] = st.number_input(
                        col, min_value=1.0, value=max(base, 1.0), key=f"in_{col}"
                    )
                else:
                    values[col] = st.number_input(col, value=base, key=f"in_{col}")
        submitted = st.form_submit_button("Predict HRI value")
    if submitted:
        X = pd.DataFrame([values], columns=feature_cols)
        pred = float(model.predict(X)[0])
        st.success(f"Predicted HRI value: {pred:.2f}")


def csv_prediction(model, feature_cols: list[str], max_rows: int) -> None:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "<div class='custom-subtle'>Upload a CSV with the same feature columns used in training.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if uploaded is None:
        return
    df = pd.read_csv(uploaded)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    preds = model.predict(df[feature_cols])
    out = df.copy()
    out["predicted_hri_value"] = preds
    st.dataframe(out.head(max_rows), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions_member1.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="HRI predictor — synthetic train / real test",
        layout="wide",
    )
    root = project_root()
    ui = sidebar_controls()
    apply_custom_style(ui["theme_mode"])

    st.title("Weekly HRI prediction (heat-affected regions)")
    st.caption(
        "Train on synthetic data; evaluate on real holdout. "
        "Target column: `hri_value`. See `data/dataset_schema.txt`."
    )

    synth_path = root / SYNTHETIC_CSV
    real_path = root / REAL_CSV
    model_path = root / MODEL_PATH
    metrics_path = root / METRICS_PATH

    tab_overview, tab_manual, tab_batch = st.tabs(
        ["Overview", "Manual Prediction", "Batch Prediction"]
    )

    with tab_overview:
        st.subheader("Data + status")
        c1, c2 = st.columns(2)
        synth_rows = len(pd.read_csv(synth_path)) if synth_path.exists() else None
        real_rows = len(pd.read_csv(real_path)) if real_path.exists() else None
        c1.metric("Synthetic train rows", str(synth_rows) if synth_rows is not None else "—")
        c2.metric("Real evaluation rows", str(real_rows) if real_rows is not None else "—")

        st.markdown(
            "**Alignment check:** run `python scripts/compare_synthetic_vs_real.py` "
            "to write `models/synthetic_vs_real_summary.json`."
        )

        metric_card(metrics_path)

        if not model_path.exists():
            st.warning(
                f"No model at `{MODEL_PATH}` yet. After training on `{SYNTHETIC_CSV}`, "
                "save your estimator there for predictions."
            )
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
            st.info("Train and save a model first to enable predictions.")
        with tab_batch:
            st.info("Train and save a model first to enable predictions.")
        return

    model = load_model(model_path)
    cols = feature_columns(root)
    with tab_manual:
        manual_prediction(model, cols)
    with tab_batch:
        csv_prediction(model, cols, ui["max_batch_rows"])


if __name__ == "__main__":
    main()
