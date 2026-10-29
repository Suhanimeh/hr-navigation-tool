# Interview questions and answers

Numbers below are from `results/SUMMARY.md` (seed 42). Re-check them against your own run before quoting.

**1. Why didn't you use accuracy?**
The positive class is 16% (attrition), 8.5% (promotion) and 31% (hiring). A model that always predicts "no" is already 84%, 91.5% and 69% accurate. On the original attrition split the "best by accuracy" model scored 87.1% against 86.7% for always-"No", with only 18% recall for leavers. I select on PR-AUC and report recall, precision and F1 for the positive class.

**2. Why PR-AUC instead of ROC-AUC?**
With a rare positive class, ROC-AUC can look good because the huge negative class keeps the false-positive *rate* small. PR-AUC reflects precision, which is what an HR reviewer actually experiences (how many flagged people are real?). Its no-skill value is the prevalence (0.16, 0.085, 0.31), so it's easy to judge improvement.

**3. What data leakage did you find and fix?**
(a) Model selection on the test set. (b) Encoders fitted on all data before splitting. (c) Shortlists scored in-sample: promotion RF had AUC 0.976 on its own training rows vs 0.792 on test; recruitment RF 1.000 vs 0.926. (d) `employee_id` as a feature. Fix: a scikit-learn `Pipeline` so preprocessing is fitted inside each CV fold, selection by CV on the train split only, out-of-fold scores for ranking, and identifiers excluded in config. A test checks that the scaler's mean equals the *training* mean.

**4. How did you handle missing values and imbalance?**
The original `dropna()` removed 15.4% of promotion rows, and the missingness was informative: every missing `previous_year_rating` was a first-year employee. I impute (median / most frequent) with a missing-indicator, inside the pipeline. For imbalance I used class weights rather than SMOTE: it's simple, can't leak synthetic points into validation folds, and PR-AUC is threshold-free so the evaluation doesn't depend on it.

**5. Which model did you pick, and are you sure it's the best?**
By CV PR-AUC: Logistic Regression (attrition), Gradient Boosting (promotion), Random Forest (hiring). I'm *not* sure for attrition and hiring: the test sets have 294 and 300 rows (~47 positives for attrition), differences of a few points are within noise, and on the hiring test set a Decision Tree is nominally ahead. Next step: repeated CV with confidence intervals. A simple, interpretable model winning on attrition isn't surprising with ~1,200 training rows.

**6. How do you evaluate fairness here?**
Sensitive columns (`Gender`, `Age`, `MaritalStatus`) are excluded by default; an ablation shows adding them changes test PR-AUC by ≤ 0.013, so the models aren't leaning on them much. Because proxies (department, region) can still encode them, I also compare selection rate, TPR and FPR by group. Disparity ratios are 0.86 (attrition/Gender), 0.98 (promotion/gender), 0.84 (hiring/Gender), all above the 0.8 rule of thumb, but base rates differ by group and these are diagnostics, not proof of fairness. In Ontario, age, sex and marital status are protected grounds, so I'd involve legal/HR before any real use.

**7. Why did you remove the "fire the highest-risk employee" feature?**
The model has AUC ≈ 0.78 on test with precision 0.37 at 0.5, so most people it flags are false alarms; it can't say when someone may leave; and it was scored in-sample. Using that to terminate someone is both methodologically unsound and harmful. Predictions are decision-support: a prompt for a human conversation. The new team-level view only shows aggregates.

**8. How is this relevant to predictive maintenance?**
The workflow transfers: baseline → benchmark → imbalance-aware metrics → error analysis → interpretability → documentation. The data doesn't: sensor time series need temporal, per-machine splits (random splits leak), often a remaining-useful-life target, cost-based thresholds, and domain knowledge of failure modes. I'd start with NASA C-MAPSS to practise that.

**9. How did you make it reproducible?**
One seed in `config.py`, stratified split, one command (`scripts/run_experiments.py`) that writes tables, plots and `run_metadata.json` (versions, features, seed), pinned minimum versions, 41 tests, and a CI workflow (written but not yet run on GitHub).

**10. What would you do with more time?**
Repeated stratified CV with confidence intervals, probability calibration, cost-based threshold selection (what does a missed leaver cost vs an unnecessary retention conversation?), SHAP for local explanations, a temporal validation if timestamps existed, and documenting the recruitment dataset's provenance and codes.
