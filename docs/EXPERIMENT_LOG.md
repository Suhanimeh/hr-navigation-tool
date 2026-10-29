# Experiment log

Add one row per run of `scripts/run_experiments.py`. Never edit old rows. `results/<task>/run_metadata.json` has the exact settings.

| Date | Task | Change vs previous | Seed | Selected (CV PR-AUC) | CV PR-AUC | Test PR-AUC | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-29 | attrition | first leakage-free pipeline | 42 | Logistic Regression | 0.606 | 0.568 | test n=294; GB 0.551 on test is within noise |
| 2026-10-29 | promotion | first leakage-free pipeline | 42 | Gradient Boosting | 0.524 | 0.533 | imputation instead of dropna |
| 2026-10-29 | recruitment | first leakage-free pipeline | 42 | Random Forest | 0.858 | 0.857 | test n=300; DT 0.862 on test; RecruitmentStrategy dominates importance |

Ideas queue: repeated stratified CV for confidence intervals; calibration curve; cost-based threshold; drop `RecruitmentStrategy` as a sensitivity test.
