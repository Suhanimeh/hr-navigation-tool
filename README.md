# HR Navigator Tool: HR Analytics & Decision Support

Three small, reproducible machine-learning case studies on public HR datasets, each with a Streamlit dashboard:

| Task | Question | Positive class | Dashboard |
|---|---|---|---|
| Attrition | Which employee profiles resemble past leavers? | `Attrition == "Yes"` | `employee-attrition-model/app.py` |
| Promotion | Which employees resemble past promotions? | `is_promoted == 1` | `employee-promotion-model/app.py` |
| Hiring | Which candidates resemble past hires? | `HiringDecision == 1` | `employee-recruitment-model/app.py` |

> **Responsible use.** These models are *decision-support* demos trained on public/fictional data. They estimate statistical association, not individual outcomes, and must never be the sole basis for hiring, firing or promotion. Results here do not establish real-world performance.

## Research objectives
1. Build leakage-free, reproducible classification pipelines for imbalanced HR data.
2. Benchmark several model families against a naive baseline with metrics suited to imbalance (PR-AUC, recall, precision, F1; not accuracy alone).
3. Select models by cross-validation on the training split only, and report the held-out test split once.
4. Analyse errors, feature reliance and group-level disparities, and state the limitations.

## Quick start
Tested with Python 3.13 (pandas 3.0.5, scikit-learn 1.9.1, streamlit 1.66.1). Other versions are untested.
```bash
pip install -r requirements.txt
# put attrition.csv, promotion.csv, recruitment.csv in data/   (see data/README.md)
python scripts/run_experiments.py            # -> results/<task>/ and results/SUMMARY.md
python -m pytest -q                           # 41 tests, synthetic data, no CSV needed
streamlit run employee-attrition-model/app.py # or the promotion / recruitment app
```
The promotion app trains on 54k rows; the first load of the *Model comparison* page takes roughly 40 s (cached afterwards).

## Project structure
```
src/hr_analytics/   config.py (targets, excluded/sensitive columns, seed)  data.py (load/validate/split)
                    modeling.py (pipelines)  evaluation.py (metrics, CV, errors, fairness)  ui.py (Streamlit helpers)
scripts/run_experiments.py   the whole experiment protocol, one command
employee-*-model/app.py      three dashboards (thin; all logic is in src/)
tests/                       pipeline tests + app smoke tests
results/                     generated outputs from the run reported below
docs/                        AUDIT, REPORT_OUTLINE, INTERVIEW_QA, APPLIED_AI_CONNECTION, EXPERIMENT_LOG
```

## Method
* **Split:** stratified 80/20, `random_state=42` (configurable with `--seed`).
* **Preprocessing (inside a scikit-learn `Pipeline`, fitted on training folds only):** median imputation + missing-indicator and scaling for numerics (scaling only for the linear model); most-frequent imputation + one-hot (`handle_unknown="ignore"`) for categoricals. Rows are **not** dropped for missing feature values.
* **Excluded columns:** identifiers (`EmployeeNumber`, `employee_id`), `PersonalityScore` (your original choice), and sensitive attributes (`Gender`, `MaritalStatus`, `Age`) by default; an ablation adds them back.
* **Models:** majority-class baseline, Logistic Regression, Decision Tree (depth 5), Random Forest, Gradient Boosting (HistGradientBoosting). Imbalance handled with class weights. SVM was dropped: O(n²) training is impractical on 54k rows and it gave no benefit to justify it.
* **Selection:** 5-fold stratified CV on the training split, by PR-AUC (the right metric when the positive class is rare). The test set is never used to choose.
* **Diagnostics:** confusion matrix, ROC and PR curves, threshold trade-off table, permutation importance, most-confident errors, per-group selection rate/TPR/FPR, sensitive-feature ablation.

## Results (from `scripts/run_experiments.py`, seed 42; full tables in `results/SUMMARY.md`)

Held-out test split, threshold 0.5. "Baseline" is the always-negative classifier.

| Task (test n) | Selected by CV | Test PR-AUC | Baseline PR-AUC | ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|---|
| Attrition (294, ~47 positives) | Logistic Regression | 0.568 | 0.160 | 0.784 | 0.681 | 0.372 | 0.765 |
| Promotion (10,962) | Gradient Boosting | 0.533 | 0.085 | 0.815 | 0.618 | 0.278 | 0.830 |
| Hiring (300) | Random Forest | 0.857 | 0.310 | 0.904 | 0.796 | 0.771 | 0.863 |

Reading these honestly:
* **Accuracy is misleading here.** The always-negative baseline scores 84% / 91.5% / 69% accuracy on the three tasks. The attrition model's 76.5% accuracy is *lower* than that baseline, yet it finds 68% of leavers, which is the point of the model.
* **Small test sets are noisy.** On attrition and hiring, models differ by a few PR-AUC points on test, which is within noise (the CV ranking and test ranking disagree for hiring: Decision Tree 0.862 vs Random Forest 0.857). Don't over-read the "winner".
* **Precision is low for attrition and promotion** (0.37 and 0.28): most flagged people are false alarms. That is why outputs are framed as prompts for human review.
* **Fairness diagnostic (test, selected model):** disparity ratio (group selection rate / highest) is 0.86 for attrition by Gender, 0.98 for promotion by gender, 0.84 for hiring by Gender (coding of 0/1 is not documented). All are above the 0.8 rule-of-thumb flag but are neither proof of fairness nor adjusted for differing base rates. Adding the sensitive columns back changed test PR-AUC by ≤ 0.013.
* **Hiring:** `RecruitmentStrategy` dominates permutation importance (≈0.40 PR-AUC drop). Because the dataset's provenance and code meanings are undocumented, treat this as a data artefact to investigate, not an insight.

### Before vs after (verified by re-running the original logic on the same files)
| Original behaviour | Measured |
|---|---|
| Attrition: model chosen by test accuracy | Best-accuracy model (LR, 87.1%) barely beats always-"No" (86.7%) on that split; its recall for leavers is 0.18 |
| Attrition: "weighted avg" recall reported as recall | Equals accuracy by construction (0.871 = 0.871), so it hides minority-class recall |
| Promotion: `dropna()` on features | Removes 8,428 of 54,808 rows (15.4%): 49% lacked a previous-year rating (all of those are first-year employees), the rest lacked education or training score |
| Promotion: shortlist scored in-sample | Random Forest AUC 0.976 on training rows vs 0.792 on test: the ranking reflects memorisation |
| Hiring: shortlist scored in-sample | Random Forest AUC 1.000 on training rows vs 0.926 on test |

The new apps score people with **out-of-fold** predictions (each person scored by a model that never saw them).

## Dashboards
Sidebar navigation: Dataset overview → Exploratory analysis → Model comparison (CV and test tables, confusion matrix, PR curves, threshold table, importance, fairness diagnostic) → a decision-support page. Every page carries the responsible-use banner, and bad/missing datasets give a clear message instead of a traceback. All 12 pages were exercised with Streamlit's `AppTest` on the real files and on synthetic data. No screenshots are included; take your own from a running app.

## Limitations & ethics
* Public or fictional datasets; no temporal split, so no evidence of performance on future data.
* The attrition data has no timestamps, so the model can't say *when* someone may leave (the original "30/60 days" wording was removed).
* Labels in the promotion and hiring data are past human decisions; models inherit their biases. Removing a sensitive column does not remove proxies (region, department).
* Age, sex and marital status are protected grounds under the Ontario Human Rights Code; using them in employment decisions raises legal risk. Not legal advice.
* Metrics are at a fixed 0.5 threshold; real use would need thresholds set from the cost of false positives vs false negatives.

## Future work
Probability calibration; cost-sensitive threshold selection; repeated CV with confidence intervals; SHAP for local explanations; document the recruitment dataset's provenance and codes.
