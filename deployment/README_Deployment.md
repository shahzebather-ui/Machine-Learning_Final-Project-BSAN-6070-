# Deployment Guide (Streamlit)

Deployment-specific notes live under `deployment/` to keep the repo root tidy.

## Local run

1. Open a terminal in the project root.
2. Install dependencies: `pip install -r requirements.txt`
3. Train a model on `data/synthetic_hri_dataset_fixed.csv` and save it (for example) to `models/member1_decision_tree.pkl`, with metrics JSON if you want the dashboard cards.
4. Run: `streamlit run app/streamlit_app.py`

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a Streamlit app and point the entry file to `app/streamlit_app.py`.
3. Cloud installs from root `requirements.txt`.
4. Ensure any required `.pkl` is in `models/` if you want live predictions in the app.

## Required files for a full prediction UI

- `app/streamlit_app.py`
- `models/member1_decision_tree.pkl` (or update the model filename in the app)
- `requirements.txt`
