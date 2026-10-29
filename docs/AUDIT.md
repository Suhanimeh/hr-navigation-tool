# Audit of the original HR Navigation Tool

Scope: the three `app.py` files (attrition 244 lines, promotion 228, recruitment 265), their `requirements.txt` files, and the git history inside the ZIP. All line numbers refer to the **original** files. "Verified" means I ran or read it; the datasets were recovered from your git history, so the numbers come from your real files.

The ZIP also contains an unrelated `image filtering and captioning` project. It was not audited.

## What existed
* Three independent Streamlit apps, each: upload CSV → decision page → EDA → model evaluation. Models: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting (+ SVM in two apps).
* No HR README (the only README describes the image apps), no tests, no datasets in the working tree, no random seeds on the models, no shared code.
* No API keys, passwords or tokens found (verified by search).

## Findings, by priority

### P1: correctness / harm
| # | File (line) | Problem | Evidence | Status |
|---|---|---|---|---|
| 1 | attrition 131, 138-152 | The app recommends firing: "Fire the Highest Risk Employee" and a PDF "HR Report – Fired Employee … should be considered for termination". Using a ~0.6 AUC model, scored in-sample, to name someone for termination is unsafe and indefensible. | Original attrition AUC 0.61-0.68 on the test split | **Removed.** Replaced by an aggregate team-level view; no individual "highest-risk" list. |
| 2 | attrition 112-117 | "May leave within ~30/60 days" has no basis; the dataset has no time-to-event information. | Columns checked | **Removed.** |
| 3 | attrition 101, promotion 85-89, recruitment (no selection) | Model "best" chosen by **test-set accuracy**: leaks the test set into selection, and accuracy is dominated by the majority class. | Always-"No" gets 86.7% on the original attrition split; the "best" model gets 87.1% with recall 0.18 | **Fixed:** CV PR-AUC on train only. |
| 4 | attrition 206-209 | "Precision/Recall/F1" use `weighted avg`. Weighted recall equals accuracy by definition and hides minority recall. | 0.871 = 0.871 (verified) | **Fixed:** positive-class metrics + macro F1. |
| 5 | promotion 94-105; recruitment 173-193 | Shortlists score people **who were in the training data** (in-sample). Trees/forests memorise them. | Promotion RF AUC 0.976 in-sample vs 0.792 test; recruitment RF 1.000 vs 0.926 | **Fixed:** out-of-fold scores. |
| 6 | promotion 32-37 | `employee_id` used as a feature. | – | **Fixed:** identifiers excluded in config. |
| 7 | promotion 57, attrition 49, recruitment 38 | `dropna()` on all columns, `inplace=True` on the shared session frame. Promotion loses 8,428 / 54,808 rows (15.4%); missingness is informative (all missing ratings are first-year employees). | Verified | **Fixed:** impute + missing-indicator in the pipeline. |

### P2: methodology
| # | Problem | Status |
|---|---|---|
| 8 | Encoders/get_dummies fitted on the full dataset before the split (attrition 67, promotion 46, recruitment 44). Mild for one-hot, but the pattern leaks. | Fixed: everything in a `Pipeline`. |
| 9 | Plain `train_test_split` with no stratification, no CV, no seeds on models → results change on every run. | Fixed: stratified, seeded, 5-fold CV. |
| 10 | No baseline model; imbalanced targets (16% / 8.5% / 31% positive) with no class weighting. | Fixed. |
| 11 | Attrition model uses only 6 of 23 usable columns (and drops `StockOptionLevel` unexplained). | Fixed: all non-excluded columns; the choice is now in `config.py`. |
| 12 | Sensitive attributes (`Gender`, `Age`, `MaritalStatus`) used as features with no discussion. | Fixed: excluded by default, ablation + group diagnostic added. |
| 13 | SVM with `probability=True` is O(n²) and was in the Streamlit rerun path. | Dropped from candidates. |

### P2: engineering
| # | File (line) | Problem | Status |
|---|---|---|---|
| 14 | attrition 94-105, 189-209; promotion 82-86, 168-195; recruitment 235 | Every model is **re-trained on every Streamlit rerun** (each slider or selectbox change on the decision page). | Fixed: `st.cache_resource`. |
| 15 | attrition 134-136 | Hard-coded `"IT"` department. The data has Sales / Research & Development / Human Resources, so this branch is dead code; it also showed `.head(5)` rows labelled "High Risk Employees". | Verified; removed. |
| 16 | recruitment 163-170 | Writes/reads `trained_model.pkl` in the working directory; "Select Best Candidates" crashes if "Train Model" wasn't clicked; pickles execute code on load. | Fixed: no pickles; `*.pkl` is git-ignored. |
| 17 | recruitment 203-209 | Nested `st.button` navigation with state stored in two different ways across the three apps. | Fixed: sidebar radio. |
| 18 | `requirements.txt` ×3 | Unpinned; `fpdf` is unmaintained and no longer needed; the three files differ. | Fixed: one file. |
| 19 | – | No tests. | Fixed: 41 tests (below). |
| 20 | attrition 40, promotion 57 | No input validation: a CSV with other columns → `KeyError` traceback. | Fixed: `validate()` + friendly messages. |

### P3: repository / portfolio
* **Repo hygiene (action needed from you).** The ZIP's git repo is rooted *inside* `employee-promotion-model/` and its remote is `suhanimehta12/HR-Navigator-Tool`. History is noisy: "update", "Delete … directory", files added/deleted repeatedly. Commit the new structure at the repository root with meaningful messages.
* **Datasets live in git history** (IBM CSV, 54,808-row promotion CSV, recruitment CSV were committed then deleted). They aren't secrets, but check their licenses. Re-adding them is blocked by `.gitignore`. To purge history use `git filter-repo`, but only if you care; it rewrites history.
* **No README describing the HR project.** Fixed (README.md).
* **Provenance:** the recruitment dataset's source is undocumented. Add it.

## Things I did not verify
* Whether the remote GitHub repo matches this ZIP.
* The CI workflow (`.github/workflows/ci.yml`) has **not been run on GitHub**; tests were run locally only.
* Behaviour under pandas 2.x / older scikit-learn (untested).
* The meaning of the 0/1 `Gender` and 1/2/3 `RecruitmentStrategy` codes in the recruitment file.

## Test evidence (run locally, Python 3.13)
`python -m pytest -q` → 41 passed. This covers loading, validation, feature/target separation, leakage (scaler fitted on train only), imputation, unseen categories, reproducibility, metrics, and rendering of every app page. The apps' 12 pages were also run against the real datasets with `AppTest`.
