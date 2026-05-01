# Model Testing Log (Member 1)

## Objective
Compare candidate regression models for `target_hri_rate` and check overfitting risk.

## Data + Split
- Dataset: `data/dataset_finalized_region9_weekly_8features.csv`
- Target: `target_hri_rate`
- Features: 8 finalized predictors (weather + socioeconomic)
- Split: `train_test_split(test_size=0.2, random_state=42)`
- Rows: train=127, test=32

## Model Runs

### Run 1: ElasticNet (baseline)
- Artifact: `models/member1_elasticnet.pkl`
- Test metrics:
  - MAE = 163.60
  - RMSE = 251.25
  - R2 = 0.460

### Run 2: DecisionTreeRegressor (current Member 1 choice)
- Params: `max_depth=5`, `min_samples_leaf=4`, `random_state=42`
- Artifact: `models/member1_decision_tree.pkl`
- Test metrics:
  - MAE = 155.86
  - RMSE = 212.07
  - R2 = 0.616

## Overfitting Check (Train vs Test)
Source: `models/overfitting_check_report.json`

### ElasticNet
- Train R2 = 0.639
- Test R2 = 0.460
- R2 gap (train - test) = 0.179
- RMSE ratio (test/train) = 1.85

### Decision Tree
- Train R2 = 0.887
- Test R2 = 0.616
- R2 gap (train - test) = 0.271
- RMSE ratio (test/train) = 2.78

## Interpretation
- Both models are usable and predictive on holdout data.
- Decision Tree performs better on test MAE/RMSE/R2 than ElasticNet.
- Decision Tree shows stronger overfitting signal (larger train-test gap), but still has better test performance.
- Keep Decision Tree as Member 1 model; mention overfitting tradeoff in presentation.

## Notes for Presentation
- We did not rely on a single run blindly; we compared two regression models.
- We tracked train and test metrics to evaluate generalization.
- Final selection balanced predictive performance and model behavior.
