# Project audit

## Scope and repository layout

The linked repository was cloned and inspected before changing its application. The project has two levels: the repository root contains `README.md` and `CreditRisk_AI.ipynb`; the runnable application is in `credit-risk-ai/` with its own `README.md`, `app.py`, `requirements.txt`, and `credit_risk_model.pkl` (about 2.8 MB). There is no dataset file, `.gitignore`, automated test suite, or other Python source file. The two README files substantially duplicate each other.

## Existing architecture and functionality

- The notebook downloads the Statlog German Credit dataset from UCI at runtime (1,000 rows, 20 predictors plus target; 700 good and 300 bad labels).
- It maps good credit to 0 and bad credit to 1, engineers three ratios, splits 80/20 with stratification and `random_state=42`, imputes/scales numerical columns, and one-hot encodes categorical columns.
- It trains balanced Logistic Regression and Random Forest pipelines. The notebook saves the Random Forest as `credit_risk_model.pkl`; recorded metadata says scikit-learn 1.6.1.
- The Streamlit app loads that Random Forest, accepts a manually assembled applicant row, shows class probability and a score/category, plots probabilities, and varies credit amount in a what-if control.
- The notebook imports SHAP and uses `TreeExplainer` for a Random Forest global summary plot. It does not provide an individual applicant explanation.
- Requirements list Streamlit, pandas, NumPy, scikit-learn, matplotlib, and joblib; the notebook also uses seaborn and SHAP, which are absent from that list.

## Existing baseline results

The notebook's saved output for its single 80/20 split reports:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7400 | 0.5465 | 0.7833 | 0.6438 | 0.8092 |
| Random Forest | 0.7800 | 0.7500 | 0.4000 | 0.5217 | 0.7973 |

These are the original recorded results and remain documented unchanged. The source data is now bundled to support later reproduction with the documented compatible environment; this update did not rerun the original notebook metrics.

## Dataset inspection completed for the upgrade

The source-format UCI data was downloaded and inspected before changing preprocessing, then retained at `credit-risk-ai/data/german.data`. Inspection found 1,000 rows, 20 predictors plus target, 700 source label 1 (good) and 300 source label 2 (bad), zero missing cells, zero exact duplicate rows, 7 numeric predictors, and 13 categorical predictors. The full source-field definitions, observed code sets, outlier scan, and limitations are in `credit-risk-ai/DATASET.md`.

Sensitive/potentially sensitive fields are age, `personal_status`, and `foreign_worker`. The personal-status field combines sex and marital status; it is not an independently encoded sex field. The source coding documents A95, but it does not appear in the inspected records. Foreign-worker status has 963/37 counts. The new applicant model excludes all three fields and preserves them in a separate audit pathway.

## Working components

- Notebook outputs capture data inspection, both model metric sets, confusion matrix/classification report, ROC curve, RF feature importance, and a SHAP summary.
- The model artifact exists and corresponds to the Random Forest pipeline according to notebook provenance.
- The app includes credit amount what-if analysis and a clear educational disclaimer.

## Missing components

- No fairness audit, group metrics, or mitigation experiment.
- No applicant-level SHAP explanation, understandable report download, or accessible description of score limitations.
- No included data file or offline route to calculate fairness metrics.
- No model artifact compatibility check or dependable path resolution.
- The app is Random Forest based; Logistic Regression is in the notebook only.

## Bugs and technical debt

- `app.py` loads `credit_risk_model.pkl` relative to the process working directory, so launching from another directory can fail.
- The app assumes `predict_proba(...)[0][1]` is the bad-credit probability instead of looking up the fitted class label.
- The app's form includes protected/demographic or sensitive proxy fields (age, personal status, foreign worker) in model inputs; these attributes can affect individual predictions. The feature list and engineered fields also must stay aligned with the serialized pipeline.
- UI options expose opaque German dataset codes without descriptions.
- Probability/risk-score labels can be read as decision guidance despite the disclaimer; risk categories are arbitrary demonstration thresholds.
- The notebook suppresses warnings globally, fetches the dataset from an external URL, installs packages in a notebook cell, and includes Google Colab-only download code.
- SHAP is missing from `requirements.txt`; seaborn is required by the notebook but also missing.
- Dependency versions are unpinned, while the pickle requires compatible Python/scikit-learn versions. The notebook records scikit-learn 1.6.1; Python compatibility is not pinned or documented.
- Duplicate README files can drift. No tests or reproducible app-level evaluation are included.

## Proposed modifications

1. Bring the existing app, model artifact, notebook, and documentation into the workspace root while preserving the original notebook and pickle.
2. Keep the original artifact and baseline results for provenance, but fit the active applicant Random Forest without sensitive fields; add a separate fairness pathway for group comparisons.
3. Add a reweighing experiment for model fitting and compare performance/fairness with the unmitigated baseline; do not describe it as a guarantee of fairness.
4. Add individual SHAP explanation support and a plain-language assessment report, plus a broader financial-attribute what-if experience.
5. Resolve paths from the app location, determine positive-class probability from model classes, document compatible dependencies/Python, and strengthen educational-use messaging.
6. Bundle and attribute the inspected dataset so the app works offline, and support a compatible user CSV for audit analysis.

## Recommended implementation sequence

1. Inspect source, notebook outputs, serialized-model provenance, and requirements. (Completed.)
2. Record this audit and preserve the notebook's original baseline figures. (Completed.)
3. Inspect and document the actual dataset before changing feature handling. (Completed.)
4. Implement the app upgrades, reusable preprocessing pipeline, and data documentation. (Completed.)
5. Check syntax and available local verification without modifying the original baseline results. (Completed.)

## Implementation completed

### Applicant and explanation upgrade

- Added modular Streamlit views in credit-risk-ai/src/faircredit_app/: overview, financial-only assessment, individual explanations, what-if simulator, aggregate fairness results, model diagnostics, and PDF export.
- Added global SHAP bar and beeswarm plots, plus individual waterfall contributions, using the fitted preprocessing pipeline and source-readable field/category labels.
- Added validation for schema fields, numeric ranges, categorical values, and rejection of demographic/target fields. What-if analysis edits amount, duration, and employment duration and recomputes the engineered amount/duration proxy.
- Added an in-memory PDF report with assessment identifier/date, predicted historical label and probability, SHAP factors, limitations, and disclaimer; no identity fields are collected or saved.
- Added METHODOLOGY.md, requirements-dev.txt, and tests covering data, preprocessing, features, training, predictions, fairness, SHAP, scenario updates, and PDF generation.
- Verification: Python 3.12 compilation passed; 11 tests passed and the SHAP integration test was skipped because Windows Application Control blocked Numba's native DLL. Streamlit AppTest rendered all seven views without app exceptions. Original notebook metrics and saved experiment results were retained.

- Kept the original notebook and Random Forest pickle at their repository paths; retained the baseline results above unchanged. The pickle is no longer used by the applicant explorer because its feature set included sensitive fields.
- Bundled the inspected UCI data and documented its values, counts, features, encodings, missingness, duplicates, outliers, sensitive attributes, and limitations.
- Added `credit-risk-ai/faircredit.py` with reusable validation, target normalization, duplicate handling, training-only imputers/scaling/one-hot encoding in `ColumnTransformer`/`Pipeline`, and stratified splitting.
- The current applicant model excludes age, personal status, and foreign-worker status. Group auditing retains these attributes separately and uses them only for evaluation and reweighing weights; proxy effects can remain.
- Reworked the Streamlit app with SHAP explanations, what-ifs, and reports. Added Fairlearn/XGBoost experiment tooling, pinned dependencies, and Python 3.10?3.12 setup notes.
- Ran actual three-model, reweighing, ExponentiatedGradient, and threshold-sensitivity experiments for age bands, personal status, and foreign-worker status. Reports and plots are in `credit-risk-ai/reports/experiments/`.
- Verification completed: Python 3.12 syntax compilation and `git diff --check` pass; all nine model/group experiment runs completed. The Streamlit UI itself was not launched.
