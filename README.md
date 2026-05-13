# BSAN 6070 — Machine Learning Final Project

**Objective:** Predict weekly **heat-related illness (HRI)** rate using weather and socioeconomic inputs. Models are **trained** on a synthetic dataset and **evaluated** on a real holdout dataset, per course direction.

## Data

| File | Use |
|------|-----|
| `data/synthetic_hri_dataset_fixed.csv` | Training set only |
| `data/final_hri_modeling_dataset.csv` | Test / evaluation set only |
| `data/dataset_schema.txt` | Column definitions |

**Target variable:** `hri_value`

##(Member 1 — Decision Tree)

```bash
pip install -r requirements.txt
```

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2
```

Outputs: `models/member1_decision_tree.pkl`, `models/member1_decision_tree_metrics.json`, `models/member1_training_runs_log.csv`

## Application

```bash
streamlit run app/streamlit_app.py
```

## Other utilities

- `scripts/compare_synthetic_vs_real.py` — distribution summary (synthetic vs real)  
- `scripts/predict_from_file.py` — batch predictions from CSV  
- `scripts/shap_member1_dt.py` — SHAP plots (requires dependencies in `requirements.txt`)

## Deployment

See `deployment/README_Deployment.md`.
