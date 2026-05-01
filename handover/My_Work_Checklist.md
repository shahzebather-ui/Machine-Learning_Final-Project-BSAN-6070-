# My Work Checklist (Member 1)

## Setup and Scope (Updated Instructions)
- [x] Confirm project scope: Region 9 weekly HRI prediction
- [x] Confirm final feature list is frozen (8 features)
- [x] Confirm no homeless variables are used
- [ ] Share scope and dataset path with team
- [ ] Confirm one primary predictive question + max two related secondary questions in final deck/report

## Data and EDA
- [x] Load finalized dataset from `data/dataset_finalized_region9_weekly_8features.csv`
- [x] Check shape, data types, and column names
- [x] Verify missing values (must be zero or handled)
- [x] Review distributions for target and key features
- [x] Run correlation check and note highly correlated pairs
- [x] Save 2-3 EDA charts for presentation

## Modeling (My Algorithm - Decision Tree)
- [x] Create train/test split strategy (consistent and documented)
- [x] Train baseline model (Decision Tree Regressor)
- [x] Evaluate MAE, RMSE, and R2
- [x] Record model assumptions and limitations
- [x] Save trained model to `models/` as `.pkl`
- [x] Save predictions and metrics table for slides
- [x] Run hyperparameter tuning (before -> tuning steps -> after)
- [x] Save tuning outputs/logs (`decision_tree_tuning_results.csv`, tuning logs, before/after visuals)
- [x] Run overfitting check (train vs test metrics + gap analysis)
- [ ] Add SHAP chart and SHAP interpretation (required in updated instructions)

## Prediction Script Requirement
- [x] Build script to read input file and generate prediction output
- [x] Ensure script loads saved model artifact correctly
- [x] Test with sample input file end-to-end
- [x] Save script in `scripts/` and document usage

## Streamlit (My Model)
- [x] Build input form for 8 features
- [x] Load my saved model and output prediction
- [x] Add simple visual (actual vs predicted or trend)
- [x] Add model metrics section (MAE/RMSE/R2)
- [x] Verify app runs locally with no errors
- [ ] Deploy on Streamlit Community Cloud and capture public app URL (required)

## Team Handoff
- [ ] Share finalized dataset and feature list with Member 2 and Member 3
- [ ] Share same split logic so model comparison is fair
- [ ] Create comparison table template for all 3 models
- [ ] Collect teammates' metrics in one final table

## Notebook + Report Compliance (New Requirement)
- [ ] Ensure notebook markdown covers rubric items (except presentation/demo + 3-page PDF + peer review)
- [ ] Add model choice justification + performance criteria justification in notebook markdown
- [ ] Add feature-selection rationale in notebook markdown
- [ ] Add limitations/future work section in notebook markdown

## GitHub + Brightspace Deliverables (New Requirement)
- [ ] Push fully executed `.ipynb`, `.py`, model artifacts, and `requirements.txt` to individual GitHub repo
- [ ] Add GitHub repo link to team report content
- [ ] Add Streamlit cloud URL to team report content
- [ ] Prepare/export presentation deck file for Brightspace upload

## Submission and Demo Readiness
- [ ] Prepare literature survey table
- [ ] Finalize feature rationale and model choice justification
- [x] Confirm deployment link/app run command works
- [ ] Practice 3-5 minute demo flow
- [ ] Final QA pass before submission
