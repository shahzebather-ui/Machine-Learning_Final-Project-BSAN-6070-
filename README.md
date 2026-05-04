# BSAN 6070 — Machine Learning Final Project

Predicting heat-related illness (**HRI**) rates using **synthetic data for training** and **real data for evaluation** only.

**Data:** `data/synthetic_hri_dataset_fixed.csv` (train), `data/final_hri_modeling_dataset.csv` (evaluate). Target: `hri_value`. Schema: `data/dataset_schema.txt`.

**Member 1 pipeline:** `scripts/train_synthetic_real_dt.py` → saves `models/member1_decision_tree.pkl` and `models/member1_decision_tree_metrics.json`.

**App:** `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`

**Deploy:** see `deployment/README_Deployment.md`.
