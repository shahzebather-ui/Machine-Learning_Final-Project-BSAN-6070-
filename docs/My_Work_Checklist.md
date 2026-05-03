# My Work Checklist (Member 1)

## Setup and Scope (Updated Instructions)
- [x] Confirm project scope: weekly HRI prediction for heat-affected regions
- [x] Confirm primary question is unchanged
- [x] Confirm dual-hypothesis framing (prediction quality + synthetic realism)
- [x] Confirm expanded feature strategy (start wide, prune by evidence)
- [x] Confirm `total_homeless` and `unsheltered` added to candidate features
- [ ] Share scope and dataset path with team
- [ ] Confirm one primary predictive question + max two related secondary questions in final deck/report

## Data and EDA
- [x] Load real dataset for holdout/evaluation
- [x] Check shape, data types, and column names
- [x] Verify missing values (must be zero or handled)
- [x] Review distributions for target and key features
- [x] Run correlation check and note highly correlated pairs
- [x] Save 2-3 EDA charts for presentation
- [ ] Lock untouched real test split before synthetic model training
- [ ] Confirm synthetic dataset target size (~2000 rows)
- [ ] Run synthetic vs real plausibility checks (ranges, distributions, outliers)

## Modeling (My Algorithm - Decision Tree)
- [x] Create baseline split strategy (consistent and documented)
- [ ] Implement synthetic-train / real-test workflow
- [ ] Train baseline model (Decision Tree Regressor) on synthetic training set
- [ ] Evaluate MAE, RMSE, and R2 on real holdout test set
- [x] Record model assumptions and limitations
- [x] Save trained model to `models/` as `.pkl`
- [x] Save predictions and metrics table for slides
- [x] Run hyperparameter tuning (before -> tuning steps -> after)
- [x] Save tuning outputs/logs (`decision_tree_tuning_results.csv`, tuning logs, before/after visuals)
- [x] Run overfitting check (train vs test metrics + gap analysis)
- [ ] Add assumption-validation evidence (not just assumptions list)
- [ ] Add SHAP chart and SHAP interpretation (required in updated instructions)

## Prediction Script Requirement
- [x] Build script to read input file and generate prediction output
- [x] Ensure script loads saved model artifact correctly
- [x] Test with sample input file end-to-end
- [x] Save script in `scripts/` and document usage

## Streamlit (My Model)
- [x] Build input form for core features
- [x] Load my saved model and output prediction
- [x] Add simple visual (actual vs predicted or trend)
- [x] Add model metrics section (MAE/RMSE/R2)
- [x] Verify app runs locally with no errors
- [ ] Add synthetic-vs-real model interpretation section
- [ ] Deploy on Streamlit Community Cloud and capture public app URL (required)

## Team Handoff
- [ ] Share finalized dataset and feature list with Member 2 and Member 3
- [ ] Share locked real-test split logic so comparison is fair
- [ ] Create comparison table template for all 3 models
- [ ] Collect teammates' metrics in one final table
- [ ] Collect synthetic data generation notes (method + quality checks) from teammate

## Notebook + Report Compliance (New Requirement)
- [ ] Ensure notebook markdown covers rubric items (except presentation/demo + 3-page PDF + peer review)
- [ ] Add model choice justification + performance criteria justification in notebook markdown
- [ ] Add feature-selection rationale in notebook markdown
- [ ] Add limitations/future work section in notebook markdown
- [ ] Add business framing section (buyer/user/application)
- [ ] Add hypothesis 2 result: can synthetic-trained model replicate real-world behavior?

## GitHub + Brightspace Deliverables (New Requirement)
- [ ] Push fully executed `.ipynb`, `.py`, model artifacts, and `requirements.txt` to individual GitHub repo
- [ ] Add GitHub repo link to team report content
- [ ] Add Streamlit cloud URL to team report content
- [ ] Prepare/export presentation deck file for Brightspace upload

## Submission and Demo Readiness
- [ ] Prepare literature survey table
- [ ] Finalize feature rationale and model choice justification
- [x] Confirm deployment link/app run command works
- [ ] Prepare one slide: model assumptions and validation checks per regressor
- [ ] Practice 3-5 minute demo flow
- [ ] Final QA pass before submission
