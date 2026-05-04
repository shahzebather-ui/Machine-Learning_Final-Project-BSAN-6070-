# Team setup — same data & rules for everyone

Use this file so no one relies on the short root `README.md` for detail.

## Install

```bash
pip install -r requirements.txt
```

## Data (everyone uses these paths)

| File | Role |
|------|------|
| `data/synthetic_hri_dataset_fixed.csv` | **Training only** (~2000 rows). Only this CSV goes into `fit()`. |
| `data/final_hri_modeling_dataset.csv` | **Evaluation only** (1460 rows). Never train on this; use it for test metrics. Includes `hri_value` and may include `regionid` (not used as a feature unless you opt in). |

**Target:** `hri_value`  
**Default features:** eight numeric columns — see `data/dataset_schema.txt`. After training, `models/member1_decision_tree_metrics.json` lists `feature_cols` (updates if Member 1 drops columns).

## Member 1 — Decision Tree (`scripts/train_synthetic_real_dt.py`)

One script for baseline, tuning, and optional feature drops:

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2
```

Feature ablation (example):

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2 --exclude-features unsheltered_homeless
```

Pick settings by **lowest `real_test_rmse`** (and sensible `real_test_r2`) in `models/member1_training_runs_log.csv` — not synthetic-train noise.

## Other scripts

| Script | Purpose |
|--------|---------|
| `scripts/compare_synthetic_vs_real.py` | `models/synthetic_vs_real_summary.json` |
| `scripts/predict_from_file.py` | Batch CSV predict; uses `feature_cols` from metrics JSON |
| `scripts/shap_member1_dt.py` | SHAP plots → `docs/analysis_outputs/` |

## Streamlit

```bash
streamlit run app/streamlit_app.py
```

Pull latest `main` before you run.
