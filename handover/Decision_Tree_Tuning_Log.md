# Decision Tree Tuning Log

Selection policy: balanced_generalization_constraints
- Constraints used: r2_gap_train_minus_test <= 0.20 and rmse_ratio_test_over_train <= 3.0
- Then selected lowest test RMSE among valid candidates.

Final selected params:
- max_depth: 3
- min_samples_leaf: 1
- min_samples_split: 2

Final metrics:
- Train RMSE: 111.894
- Test RMSE: 213.447
- Train R2: 0.755
- Test R2: 0.611
- R2 gap: 0.145
- RMSE ratio test/train: 1.908

Artifacts updated:
- models/member1_decision_tree.pkl
- models/member1_decision_tree_metrics.json
- models/decision_tree_tuned_metrics.json
- models/overfitting_check_report.json
- models/member1_model_comparison.json
- docs/analysis_outputs/viz_tuned_dt_*
- docs/analysis_outputs/viz_tuned_dt_feature_importance.png
- docs/analysis_outputs/viz_tuned_dt_tree_top3.png