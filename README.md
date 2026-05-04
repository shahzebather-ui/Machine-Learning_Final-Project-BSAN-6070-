# BSAN 6070 — Machine Learning Final Project

Predicting heat-related illness (**HRI**) rates: **train** on `data/synthetic_hri_dataset_fixed.csv` only; **evaluate** on `data/final_hri_modeling_dataset.csv` only. Target: `hri_value`. Schema: `data/dataset_schema.txt`.

**Teammates (data split + how to run models):** read **`docs/TEAM_README.md`** after you `git pull` — that is the shared handoff; you do not need a separate download.

**Member 1 (Decision Tree):** `scripts/train_synthetic_real_dt.py` → `models/member1_decision_tree.pkl` + `models/member1_decision_tree_metrics.json`.

**App:** `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`

**Deploy:** `deployment/README_Deployment.md`
