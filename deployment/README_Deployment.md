# Deployment Guide (Streamlit)

All deployment-specific files are intentionally grouped under `deployment/` to reduce confusion.

## Local Run
1. Open terminal in project root.
2. Install requirements:
   - `pip install -r requirements.txt`
3. Train model (if needed):
   - `python scripts/train_member1_model.py`
4. Run app:
   - `streamlit run app/streamlit_app.py`

## Streamlit Community Cloud
1. Push this project to GitHub.
2. Create a new Streamlit app and connect the repo/branch.
3. Set app entrypoint to:
   - `app/streamlit_app.py`
4. Streamlit Cloud will install dependencies from root `requirements.txt`.
5. Deploy and verify app loads prediction UI.

## Required Files for Cloud
- `app/streamlit_app.py`
- `models/member1_elasticnet.pkl`
- `requirements.txt` (root)
- data files only if app needs them at runtime
