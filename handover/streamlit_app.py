from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


FEATURE_COLS = [
    "feat_mean_tmax_c_week",
    "feat_max_tmax_c_week",
    "feat_temp_range_c_week",
    "feat_heat_intensity",
    "feat_poverty_rate",
    "feat_unemployment_rate",
    "feat_median_hh_income",
    "feat_total_population",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def metric_card(metrics_path: Path) -> None:
    if not metrics_path.exists():
        st.info("No metrics file found yet. Run training script first.")
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
            /* Kill default red accents */
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


def sidebar_controls(root: Path) -> dict:
    st.sidebar.header("Display Controls")
    theme_mode = st.sidebar.selectbox(
        "Theme Accent",
        ["Default Blue", "Black + Yellow", "Neon (Blue + Purple)"],
        index=0,
    )
    show_dataset_preview = st.sidebar.checkbox("Show dataset preview", value=True)
    show_extra_overview_cards = st.sidebar.checkbox("Show quick overview cards", value=True)
    show_uploaded_tree_first = st.sidebar.checkbox(
        "Use uploaded dark tree first", value=True
    )
    max_batch_rows = st.sidebar.slider("Batch preview rows", min_value=5, max_value=100, value=20, step=5)
    return {
        "theme_mode": theme_mode,
        "show_dataset_preview": show_dataset_preview,
        "show_extra_overview_cards": show_extra_overview_cards,
        "show_uploaded_tree_first": show_uploaded_tree_first,
        "max_batch_rows": max_batch_rows,
    }


def manual_prediction(model) -> None:
    st.subheader("Manual Input Prediction")
    st.markdown(
        "<div class='custom-subtle'>Enter feature values and run one prediction.</div>",
        unsafe_allow_html=True,
    )
    with st.form("manual_pred_form"):
        values = {}
        left, right = st.columns(2)
        with left:
            values["feat_mean_tmax_c_week"] = st.number_input(
                "Mean Weekly Max Temp (C)", value=30.0
            )
            values["feat_max_tmax_c_week"] = st.number_input(
                "Max Weekly Temp (C)", value=35.0
            )
            values["feat_temp_range_c_week"] = st.number_input(
                "Weekly Temp Range (C)", value=5.0
            )
            values["feat_heat_intensity"] = st.number_input(
                "Heat Intensity", value=2.0
            )
        with right:
            values["feat_poverty_rate"] = st.number_input(
                "Poverty Rate (0-1)", min_value=0.0, max_value=1.0, value=0.14
            )
            values["feat_unemployment_rate"] = st.number_input(
                "Unemployment Rate (0-1)", min_value=0.0, max_value=1.0, value=0.06
            )
            values["feat_median_hh_income"] = st.number_input(
                "Median Household Income", value=70000.0
            )
            values["feat_total_population"] = st.number_input(
                "Total Population", min_value=1.0, value=50000000.0
            )
        submitted = st.form_submit_button("Predict HRI Rate")
    if submitted:
        X = pd.DataFrame([values], columns=FEATURE_COLS)
        pred = float(model.predict(X)[0])
        st.success(f"Predicted weekly HRI rate: {pred:.2f}")


def csv_prediction(model, max_rows: int) -> None:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "<div class='custom-subtle'>Upload a CSV containing the 8 feature columns.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload CSV with the 8 feature columns", type=["csv"])
    if uploaded is None:
        return
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    preds = model.predict(df[FEATURE_COLS])
    out = df.copy()
    out["predicted_hri_rate"] = preds
    st.dataframe(out.head(max_rows), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions_member1.csv",
        mime="text/csv",
    )


def show_model_insights(root: Path, show_uploaded_tree_first: bool) -> None:
    st.subheader("Model Insights")
    st.markdown(
        "<div class='custom-subtle'>Charts below are generated during analysis/tuning.</div>",
        unsafe_allow_html=True,
    )
    viz_dir = root / "docs" / "analysis_outputs"
    candidates = []
    if show_uploaded_tree_first:
        candidates.append(
            ("Uploaded Decision Tree (Custom)", "viz_decision_tree_custom_uploaded.png")
        )
    candidates += [
        ("Tuned Decision Tree (Top Levels)", "viz_tuned_dt_tree_top3.png"),
        ("Feature Importances", "viz_tuned_dt_feature_importance.png"),
        ("Train vs Test RMSE", "viz_tuned_dt_train_test_rmse.png"),
        ("Train vs Test R²", "viz_tuned_dt_train_test_r2.png"),
        ("Residual Distribution", "viz_tuned_dt_residual_distribution.png"),
        ("Before vs After Summary", "viz_dt_before_vs_after_summary_table.png"),
    ]
    shown = 0
    for title, filename in candidates:
        p = viz_dir / filename
        if p.exists():
            st.markdown(f"**{title}**")
            st.image(str(p), use_container_width=True)
            shown += 1
    if shown == 0:
        st.info("No analysis visuals found yet. Generate charts first.")


def main() -> None:
    st.set_page_config(
        page_title="Region 9 (CA, AZ, NV, HI) Weekly HRI Predictor", layout="wide"
    )
    root = project_root()
    ui = sidebar_controls(root)
    apply_custom_style(ui["theme_mode"])
    st.title("Region 9 (CA, AZ, NV, HI) Weekly HRI Predictor")
    st.caption("Decision Tree Regressor (Member 1) with 8 finalized features")
    model_path = root / "models" / "member1_decision_tree.pkl"
    metrics_path = root / "models" / "member1_decision_tree_metrics.json"

    if not model_path.exists():
        st.warning(
            "Model not found. Run Decision Tree training/tuning first."
        )
        st.stop()

    model = load_model(model_path)
    tab_overview, tab_manual, tab_batch, tab_insights = st.tabs(
        ["Overview", "Manual Prediction", "Batch Prediction", "Model Insights"]
    )

    with tab_overview:
        st.subheader("Performance Overview")
        metric_card(metrics_path)
        if ui["show_extra_overview_cards"]:
            c1, c2, c3 = st.columns(3)
            data_path = root / "data" / "dataset_finalized_region9_weekly_8features.csv"
            tuned_path = root / "models" / "decision_tree_tuned_metrics.json"
            if data_path.exists():
                row_count = len(pd.read_csv(data_path))
                c1.metric("Rows in dataset", f"{row_count}")
            if tuned_path.exists():
                tuned = json.loads(tuned_path.read_text())
                c2.metric("Selected max_depth", str(tuned["selected_params"]["max_depth"]))
                c3.metric("Overfit R² gap", f"{tuned['overfit_signals']['r2_gap_train_minus_test']:.3f}")
        if ui["show_dataset_preview"]:
            data_path = root / "data" / "dataset_finalized_region9_weekly_8features.csv"
            if data_path.exists():
                st.markdown("**Dataset Preview**")
                st.dataframe(pd.read_csv(data_path).head(8), use_container_width=True)
        st.markdown(
            "<div class='custom-subtle'>Use tabs to run predictions or show charts in presentation mode.</div>",
            unsafe_allow_html=True,
        )
    with tab_manual:
        manual_prediction(model)
    with tab_batch:
        csv_prediction(model, ui["max_batch_rows"])
    with tab_insights:
        show_model_insights(root, ui["show_uploaded_tree_first"])


if __name__ == "__main__":
    main()
