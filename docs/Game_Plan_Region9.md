# BSAN6070 Final Project - Game Plan (Region 9 Weekly HRI)

## Project Scope (Updated Per Professor)
- Geography focus: heat-affected US regions (state/city level acceptable if tied to heat burden)
- Time level: weekly
- Target: `target_hri_rate` (heat-related illness rate per week)
- Real-world dataset: expanded from initial 160 records to about 1460 records (evaluation dataset)
- Synthetic dataset: generate around 2000 synthetic records for training
- Feature policy: start broad, then remove weak features through iterative modeling

## Main Predictive Question
Can we accurately predict weekly heat-related illness rate in heat-affected regions using weather and socioeconomic indicators?

## Business/Application Framing (Professor Requirement)
- Public health departments can use weekly risk forecasts for heat-alert planning.
- Emergency departments and local governments can use predictions for staffing and resource allocation.
- Community organizations can use risk estimates to target cooling center and outreach efforts.
- Product framing: a weekly HRI forecasting decision-support tool for heat-vulnerable regions.

## Hypotheses
- Hypothesis 1: Weather and socioeconomic indicators can predict weekly HRI rates with useful accuracy.
- Hypothesis 2: Synthetic tabular data can approximate real-world patterns enough to train models that generalize to real data.

## Initial Feature Set (Expand First, Then Select)
- `feat_mean_tmax_c_week`
- `feat_max_tmax_c_week`
- `feat_temp_range_c_week`
- `feat_heat_intensity`
- `feat_poverty_rate`
- `feat_unemployment_rate`
- `feat_median_hh_income`
- `feat_total_population`
- `feat_total_homeless`
- `feat_unsheltered`

## Success Metrics
- Primary: RMSE, MAE
- Secondary: R2
- Selection rule: lowest real-holdout RMSE with acceptable train-test gap and documented assumption checks

## Team Workflow (3 Members)

### Member 1 (You)
- Own data alignment checks, baseline model, assumption-validation notes, and Streamlit deployment
- Train and evaluate `Decision Tree Regressor` using synthetic-train and real-test setup
- Save model artifact and build prediction script from input file

### Member 2
- Lead synthetic data generation and synthetic quality checks
- Train and evaluate `Random Forest Regressor`
- Report feature importance and tuning notes

### Member 3
- Train and evaluate `XGBoost Regressor` (or `Gradient Boosting Regressor`)
- Report tuning and final metrics on same real holdout

### Group Merge Step
- Compare all 3 models on same real holdout and same metrics
- Add synthetic-vs-real findings section
- Select one final model for team demo

## Deliverables Checklist
- [ ] Business use-case statement (who uses it and why)
- [ ] Literature survey table (related prior work + what is different)
- [ ] Data quality checks (real + synthetic: missingness, ranges, outliers)
- [ ] Feature rationale (why each feature is kept/removed)
- [ ] Model assumptions and validation notes
- [ ] Tuning steps and justification
- [ ] Final comparison table (3 models)
- [ ] Synthetic-vs-real model behavior discussion
- [ ] Streamlit app working demo
- [ ] Prediction script from file input using saved model

## Immediate Action Plan (Fast Path)
1. Finalize real dataset schema and lock real test split first.
2. Generate synthetic dataset (about 2000 rows) from heat-affected regions feature space.
3. Run synthetic quality checks against real data distributions.
4. Train regressors on synthetic data only.
5. Evaluate all models on untouched real test data.
6. Document model-assumption checks per regressor.
7. Update Streamlit with model metrics and synthetic-vs-real interpretation.

## Suggested Timeline (Your Part)
- Data/schema + holdout lock: 2-3 hours
- Baseline synthetic-train/real-test modeling: 3-5 hours
- Assumption checks + comparison write-up: 2-3 hours
- Streamlit update + testing: 2-4 hours
- Total expected: 9-15 hours

