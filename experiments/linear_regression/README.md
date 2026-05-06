# Linear regression experiment (isolated from Decision Tree)

This folder is **only** for comparing **sklearn `LinearRegression`** (OLS) on the same workflow as Member 1:

- **Train:** `data/synthetic_hri_dataset_fixed.csv`
- **Test:** `data/final_hri_modeling_dataset.csv`
- **Default features:** the same **8** numeric columns as `scripts/train_synthetic_real_dt.py` (before any DT ablation drops).

Artifacts are written under **`outputs/`** here so nothing overwrites:

- `models/member1_decision_tree.pkl`
- `models/member1_decision_tree_metrics.json`

## Run (from repo root)

```bash
python3 experiments/linear_regression/train_linear_regression_synthetic_real.py
```

Default outputs:

- `experiments/linear_regression/outputs/linear_regression_8feat.pkl`
- `experiments/linear_regression/outputs/linear_regression_8feat_metrics.json`
- `experiments/linear_regression/outputs/lr_training_runs_log.csv` (append one row per run)

## Optional: same feature drops as an ablation

```bash
python3 experiments/linear_regression/train_linear_regression_synthetic_real.py \
  --exclude-features max_temp_celsius feat_unemployment_rate feat_poverty_rate overall_homeless \
  --model-out experiments/linear_regression/outputs/linear_regression_4feat.pkl \
  --metrics-out experiments/linear_regression/outputs/linear_regression_4feat_metrics.json
```

Compare **`test_on_real.rmse`** (and MAE / R²) in the metrics JSON to your frozen DT.

**Fair apples-to-apples run (same 4 features as frozen Member 1 DT):** after a default 8-feature run, we also saved  
`linear_regression_4feat_same_as_frozen_dt_metrics.json` (exclude `max_temp_celsius`, `feat_unemployment_rate`, `feat_poverty_rate`, `overall_homeless`). On a quick local run, that LR beat the frozen DT slightly on **real RMSE** while DT was lower on **MAE** — open both JSON files and align with your teammates’ metrics before any switch.

If LR wins on the metrics your team cares about, swapping the **main** app model is a deliberate change: point Streamlit + scripts at the new `.pkl` / metrics paths (not done automatically here).
