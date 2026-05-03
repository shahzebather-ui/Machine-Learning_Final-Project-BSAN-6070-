# Machine Learning — Final Project (BSAN 6070)

Heat-related illness (HRI) rate prediction for heat-affected US regions using a **synthetic training set** and a **real holdout evaluation set**, per updated course direction.

## Data (canonical)

| File | Role |
|------|------|
| `data/synthetic_hri_dataset_fixed.csv` | Training (~2000 synthetic rows) |
| `data/final_hri_modeling_dataset.csv` | Real evaluation / test (1460 rows) |

Schema notes: `data/dataset_schema.txt`.

## Scripts

- `scripts/compare_synthetic_vs_real.py` — distribution summary (JSON) for synthetic vs real alignment checks.
- `scripts/predict_from_file.py` — batch predictions once a trained `.pkl` exists under `models/`.

## App

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If `models/member1_decision_tree.pkl` is not present yet, the app still loads data previews; train on synthetic data first, then save the model to that path (or adjust paths in the app).

## Deployment

See `deployment/README_Deployment.md`.
