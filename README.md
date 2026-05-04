# Machine Learning — Final Project (BSAN 6070)

Heat-related illness (**HRI**) rate prediction for heat-affected US regions.  
**Train** only on synthetic data; **evaluate** only on real data (professor workflow).

---

## Team: start here

1. **Install** (use a venv if you use one):

   ```bash
   pip install -r requirements.txt
   ```

2. **Data you all use** (same files, same column names):

   | File | Role |
   |------|------|
   | `data/synthetic_hri_dataset_fixed.csv` | **Training** (~2000 rows) — `fit()` only on this |
   | `data/final_hri_modeling_dataset.csv` | **Test / evaluation** (1460 rows) — **never** used in `fit()`; has `hri_value` as target and may include `regionid` (not a default feature) |

3. **Target column:** `hri_value`  
4. **Feature columns (default 8):** see `data/dataset_schema.txt` and `member1_decision_tree_metrics.json` → `feature_cols` after training.

5. **Schedule:** In-person **presentation is Tuesday May 5, 6:00 PM PST** (confirm with your section). **Written / notebook / Brightspace** deadlines are **whatever the syllabus says** — do not mix them up with the presentation date.

---

## Member 1 (Decision Tree) — main training script

**There is a single current script** (not an “old” vs “new” split):

- **`scripts/train_synthetic_real_dt.py`**

It always: loads synthetic + real CSVs, fits a **sklearn** `DecisionTreeRegressor` on **synthetic only**, scores on **real only**, saves `models/member1_decision_tree.pkl` + `models/member1_decision_tree_metrics.json`, and appends a row to `models/member1_training_runs_log.csv`.

### Commands

**Baseline / tuning** (change numbers to try other trees):

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2
```

**Feature ablation (drop one or more columns for this run):**

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2 --exclude-features unsheltered_homeless
```

**“Put a feature back”** = run again **without** that column in `--exclude-features` (or with a shorter list). Each run overwrites the `.pkl` / metrics with **that** run’s settings.

**Pick the winner using `real_test_rmse` and `real_test_r2` in the log** — not synthetic train numbers.

---

## Other scripts

| Script | Purpose |
|--------|---------|
| `scripts/compare_synthetic_vs_real.py` | Writes `models/synthetic_vs_real_summary.json` (distribution check) |
| `scripts/predict_from_file.py` | Batch predict; reads **`feature_cols` from** `models/member1_decision_tree_metrics.json` (stays in sync if features are dropped) |
| `scripts/shap_member1_dt.py` | SHAP bar + beeswarm PNGs → `docs/analysis_outputs/` (needs `pip install` from requirements) |

---

## Streamlit

```bash
streamlit run app/streamlit_app.py
```

The app reads **`feature_cols` from the metrics JSON**, so it still works after optional feature drops.

---

## Git / repo

`main` includes synthetic + real CSVs, training script, and model artifacts as Member 1 commits them. Pull before you run.

---

## Deployment

See `deployment/README_Deployment.md` for Streamlit Cloud notes.
