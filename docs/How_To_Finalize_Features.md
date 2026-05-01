# How To Finalize Features (Simple Process)

Use this 5-step rule to finalize features quickly and defend your choices in class.

## Step 1: Keep relevance to target
Keep features that logically influence heat-related illness:
- weekly temperature intensity and variability
- socioeconomic vulnerability indicators

## Step 2: Remove leakage and duplicates
- Do not include anything derived from the target.
- If two features are almost the same, keep the clearer one.

## Step 3: Keep feature count small (6 to 8)
This project uses 8 predictors for speed and explainability:
- `feat_mean_tmax_c_week`
- `feat_max_tmax_c_week`
- `feat_temp_range_c_week`
- `feat_heat_intensity`
- `feat_poverty_rate`
- `feat_unemployment_rate`
- `feat_median_hh_income`
- `feat_total_population`

## Step 4: Validate statistically
- Check null values (must be zero or imputed cleanly).
- Check pairwise correlation (remove one of highly redundant pairs).
- Run baseline model and inspect coefficient signs or feature importance.

## Step 5: Freeze and share
Once metrics are stable:
- freeze the feature list,
- freeze train/test split logic,
- share the same finalized dataset with all teammates for fair model comparison.

