# Research report outline (6-8 pages)

1. **Abstract** (150 words): problem, three tasks, protocol, headline finding ("accuracy hides minority-class performance; leakage-free CV gives PR-AUC 0.57 / 0.53 / 0.86").
2. **Introduction & objectives:** why HR classification is imbalanced and high-stakes; decision-support framing.
3. **Data:** source, size, positive rate, missingness, excluded columns, license; synthetic/public caveat. Table from `results/*/run_metadata.json`.
4. **Methods**
   * 4.1 Split and CV protocol (diagram: train → 5-fold CV → select; test touched once)
   * 4.2 Preprocessing pipeline and leakage controls
   * 4.3 Models and class-imbalance handling
   * 4.4 Metrics and why PR-AUC drives selection
   * 4.5 Diagnostics: errors, importance, group analysis, ablation
5. **Results:** per task, CV table, test table, curves, confusion matrix, threshold table (`results/SUMMARY.md`).
6. **Error analysis:** most-confident false positives/negatives (`errors_*.csv`); cost of missed leavers vs false alarms; what the model can't see.
7. **Fairness and responsible use:** group table, ablation, proxy variables, legal context, why aggregate views and human review.
8. **Before/after:** what the original implementation did and measured effects (README table).
9. **Limitations & threats to validity:** public data, no temporal split, small test sets, fixed threshold, label bias, single seed.
10. **Future work:** repeated CV + CIs, calibration, cost-based thresholds, SHAP, temporal validation.
11. **Reproducibility appendix:** commands, versions, seed, file list.

**Short presentation (8 slides):** problem → data → protocol diagram → baseline vs models → why accuracy misleads → error/fairness → what I fixed → limitations & next steps.

**Experiment log:** keep `docs/EXPERIMENT_LOG.md` (template inside); one row per run.

**Resume / portfolio blurb**
> Built and audited an HR analytics decision-support system (attrition, promotion, hiring) in Python/scikit-learn/Streamlit. Replaced accuracy-based, leakage-prone model selection with a reproducible pipeline (stratified 5-fold CV, class-weighted models, PR-AUC selection, baseline comparison), found and removed in-sample scoring (AUC 0.98 train vs 0.79 test), added error analysis, permutation importance and group-fairness diagnostics, and 41 automated tests.
