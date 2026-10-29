# How this project connects to applied AI / predictive maintenance

HR analytics is **not** predictive maintenance. What transfers is the methodology.

| Research skill (job posting) | Where it appears here |
|---|---|
| Dataset preparation | Validation, imputation with missing-indicators, identifier/sensitive-column exclusion (`data.py`, `modeling.py`) |
| Selecting methods | Baseline → linear → tree → ensembles, with reasons for dropping SVM |
| Benchmarking | One protocol for all models, CV on train, single test read-out |
| Metric selection | PR-AUC / recall / precision under 8-31% positive rates; why accuracy fails |
| Error analysis | `errors_false_*.csv`, threshold table, FP vs FN cost discussion |
| Interpretability | Permutation importance (and its limits: association, not causation) |
| Experiment tracking | `run_metadata.json`, seed, versions, `EXPERIMENT_LOG.md` |
| Reporting | README, report outline, auto-generated `SUMMARY.md` |
| Responsible AI | Decision-support framing, fairness diagnostic, removal of the termination feature |

## What would be different for equipment failure prediction
* **Data:** multivariate time series from sensors (vibration, temperature, current), not one row per person. Needs windowing and feature engineering (rolling statistics, FFT/spectral features) or sequence models.
* **Labels:** failures are rarer (often <1%), may be censored, and the target may be *remaining useful life* (regression/survival) rather than a binary class.
* **Validation:** splits must be **temporal and by machine/asset**; random splits leak because neighbouring windows of the same machine are near-duplicates. This project's random split would be wrong there.
* **Costs:** a missed failure (downtime, safety) vs a false alarm (unnecessary maintenance) have measurable costs, so thresholds should be cost-driven.
* **Domain knowledge:** failure modes, maintenance logs, sensor physics, and data quality issues (drift, sensor dropout).
* **Methods to learn next:** survival analysis, anomaly detection, LSTM/1D-CNN or gradient boosting on engineered features, public benchmarks (NASA C-MAPSS turbofan, CWRU bearing).

Say this plainly in an interview: "I know the HR data is a different domain; what I can show is a rigorous evaluation workflow, and I know where it would need to change for time-series failure data."
