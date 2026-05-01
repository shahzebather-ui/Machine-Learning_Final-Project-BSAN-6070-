# BSAN6070 Final Project - Game Plan (Region 9 Weekly HRI)

## Project Scope (Locked)
- Geography level: `HHS Region 9` (CA/AZ/NV/HI combined)
- Time level: weekly
- Target: `target_hri_rate` (heat-related illness rate per week)
- Dataset: no homeless variables, reduced feature set for speed and explainability

## Main Predictive Question
Can we accurately predict weekly heat-related illness rate in Region 9 using weather and socioeconomic indicators?

## Finalized Feature Set
- `feat_mean_tmax_c_week`
- `feat_max_tmax_c_week`
- `feat_temp_range_c_week`
- `feat_heat_intensity`
- `feat_poverty_rate`
- `feat_unemployment_rate`
- `feat_median_hh_income`
- `feat_total_population`

## Success Metrics
- Primary: RMSE, MAE
- Secondary: R2
- Model selection rule: lowest RMSE with acceptable interpretability

## Team Workflow (3 Members)

### Member 1 (You)
- Own data pipeline, baseline model, and Streamlit deployment
- Train and evaluate `Decision Tree Regressor`
- Save model artifact and build prediction script from input file

### Member 2
- Train and evaluate `Random Forest Regressor`
- Report feature importance and tuning notes

### Member 3
- Train and evaluate `XGBoost` (or Gradient Boosting Regressor)
- Report tuning and final metrics

### Group Merge Step
- Compare all 3 models on same split and same metrics
- Select one final model for team demo

## Deliverables Checklist
- [ ] Literature survey table (related prior work + what is different)
- [ ] Data quality checks (missingness, ranges, outliers)
- [ ] Feature rationale (why each feature is included)
- [ ] Model assumptions/validation notes
- [ ] Tuning steps and justification
- [ ] Final comparison table (3 models)
- [ ] Streamlit app working demo
- [ ] Prediction script from file input using saved model

## Immediate Action Plan (Fast Path)
1. Run EDA on finalized dataset and confirm no null values.
2. Split data (time-aware split preferred).
3. Train baseline model (Member 1) and store metrics.
4. Build Streamlit app with:
   - manual input form
   - prediction output
   - small chart of actual vs predicted
5. Hand same dataset and split strategy to Members 2 and 3.

## Suggested Timeline (Your Part)
- Data/EDA/feature lock: 2-3 hours
- Baseline model + metrics + save `.pkl`: 2-3 hours
- Streamlit app + test run: 2-4 hours
- Total expected: 6-10 hours

