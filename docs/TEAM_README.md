# Teammate handoff — same data, your own model

This file is in the repo: **`docs/TEAM_README.md`**. After `git pull origin main`, open it here.  
Root **`README.md`** is the short summary; **this file** is the step-by-step for the group.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Data (everyone uses the same files)

| File | Role |
|------|------|
| `data/synthetic_hri_dataset_fixed.csv` | **Training only** (~2000 rows). Call `fit()` on this only. |
| `data/final_hri_modeling_dataset.csv` | **Evaluation only** (1460 rows). Never `fit()` on this. Compare predictions to **`hri_value`**. |

- **Target:** `hri_value`  
- **Default features (8):** in `data/dataset_schema.txt`. Real file may include `regionid` — not used unless you choose it.  
- **Pick hyperparameters by `real_test_rmse` / `real_test_r2`** in `models/member1_training_runs_log.csv` (Member 1 example), not by synthetic train fit alone.

---

## Member 1 script (Decision Tree) — reference only

One script: `scripts/train_synthetic_real_dt.py`

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2
```

Optional feature drop:

```bash
python3 scripts/train_synthetic_real_dt.py --max-depth 5 --min-samples-leaf 9 --min-samples-split 2 --exclude-features COLUMN_NAME
```

You **your** model: same CSV split rule; **your** `.pkl` + metrics; same comparison table row format as agreed with the team.

---

## Other utilities

| Script | Use |
|--------|-----|
| `scripts/compare_synthetic_vs_real.py` | Alignment JSON under `models/` |
| `scripts/predict_from_file.py` | Batch predict (reads `feature_cols` from metrics JSON) |
| `scripts/shap_member1_dt.py` | SHAP plots (Member 1); your model needs your own SHAP if required |

---

## Streamlit

```bash
streamlit run app/streamlit_app.py
```

Member 1’s app loads features from their metrics JSON; if you deploy your own model, adjust paths or build your own tab.

---

## Timeline

**Presentation: Tuesday May 5, 6:00 PM PST** (confirm with section).  
Other deadlines: **syllabus / Brightspace**, not assumed here.
