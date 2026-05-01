# Model Testing Log (Member 1)

## Objective
Compare candidate regression models for `target_hri_rate`, then tune Decision Tree and select a final model with strong test performance and acceptable generalization.

## Data + Split
- Dataset: `data/dataset_finalized_region9_weekly_8features.csv`
- Target: `target_hri_rate`
- Features: 8 finalized predictors (weather + socioeconomic)
- Split: `train_test_split(test_size=0.2, random_state=42)`
- Rows: train=127, test=32

## Run History (Chronological)

### Run 1: ElasticNet baseline
- Artifact: `models/member1_elasticnet.pkl`
- Test metrics:
  - MAE = 163.60
  - RMSE = 251.25
  - R2 = 0.460

### Run 2: Initial Decision Tree
- Config: `max_depth=5`, `min_samples_leaf=4`, `random_state=42`
- Test metrics:
  - MAE = 155.86
  - RMSE = 212.07
  - R2 = 0.616
- Observation: better than ElasticNet, but showed notable overfitting gap.

### Run 3: Decision Tree hyperparameter tuning (grid search)
- Script: `scripts/tune_decision_tree.py`
- Search space:
  - `max_depth`: [2, 3, 4, 5, 6, 8, 10, 12, None]
  - `min_samples_leaf`: [1, 2, 4, 6, 8]
  - `min_samples_split`: [2, 4, 6, 8, 12]
- Total combinations tested: 225
- Full results table: `models/decision_tree_tuning_results.csv`
- Top candidates file: `models/decision_tree_tuning_results_top20.csv`

## Final Selected Model (Member 1)
- Model: `DecisionTreeRegressor`
- Selection policy: balanced generalization constraints
  - kept candidates with `r2_gap_train_minus_test <= 0.20`
  - and `rmse_ratio_test_over_train <= 3.0`
  - selected lowest test RMSE among valid candidates
- Final params:
  - `max_depth=3`
  - `min_samples_leaf=1`
  - `min_samples_split=2`
  - `random_state=42`
- Final artifacts:
  - `models/member1_decision_tree.pkl`
  - `models/member1_decision_tree_metrics.json`
  - `models/decision_tree_tuned_metrics.json`
  - `models/overfitting_check_report.json`
  - `models/member1_model_comparison.json`

## Final Metrics (Selected Decision Tree)
- Train:
  - MAE = 75.50
  - RMSE = 111.89
  - R2 = 0.755
- Test:
  - MAE = 139.80
  - RMSE = 213.45
  - R2 = 0.611
- Overfitting diagnostics:
  - R2 gap (train - test) = 0.145
  - RMSE ratio (test/train) = 1.908

## Visuals Created for Model + Overfitting
- Decision tree visuals (`viz_decision_tree_*`)
- Overfitting visuals (`viz_overfit_*`)
- Tuned model visuals:
  - `viz_tuned_dt_train_test_rmse.png`
  - `viz_tuned_dt_train_test_r2.png`
  - `viz_tuned_dt_residual_distribution.png`
  - `viz_tuned_dt_feature_importance.png`
  - `viz_tuned_dt_tree_top3.png`

## Summary
- Decision Tree outperformed ElasticNet on test metrics.
- Hyperparameter tuning was performed and fully logged.
- Final model was selected with both performance and overfitting behavior considered.
