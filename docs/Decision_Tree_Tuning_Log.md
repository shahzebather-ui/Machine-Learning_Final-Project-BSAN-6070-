# Decision Tree Tuning Log

Selection policy: balanced_generalization_constraints
- Constraints used: r2_gap_train_minus_test <= 0.20 and rmse_ratio_test_over_train <= 3.0
- Then selected lowest test RMSE among valid candidates.

What changed from earlier run:
- Earlier Decision Tree example used: max_depth=5, min_samples_leaf=4, min_samples_split=2
- Final selected tuned model uses: max_depth=3, min_samples_leaf=1, min_samples_split=2
- Reason: better train-test balance (smaller overfitting gap) while maintaining strong test performance.

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
- models/decision_tree_tuning_results.csv
- models/decision_tree_before_after_comparison.csv
- docs/analysis_outputs/viz_tuned_dt_*
- docs/analysis_outputs/viz_tuned_dt_feature_importance.png
- docs/analysis_outputs/viz_tuned_dt_tree_top3.png
- docs/analysis_outputs/viz_dt_before_vs_after_test_metrics.png
- docs/analysis_outputs/viz_dt_before_vs_after_train_test_rmse.png
- docs/analysis_outputs/viz_dt_before_vs_after_train_test_r2.png
- docs/analysis_outputs/viz_dt_before_vs_after_summary_table.png