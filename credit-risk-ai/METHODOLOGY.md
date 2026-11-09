# FairCredit methodology and applicant experience

## Purpose

FairCredit is an educational research prototype using the historical UCI Statlog German Credit sample. It does not make bank decisions, provide a real credit score, predict loan approval, guarantee an outcome, or replace a professional financial assessment.

## Prediction and preprocessing

The target is normalized to class 0 = source good-credit label and class 1 = source bad-credit label. The applicant pipeline excludes age, personal status/sex code, and foreign-worker status. A stratified train/test split is made before pipeline fitting. Median imputation and scaling for numeric columns, plus mode imputation and one-hot encoding for categorical columns, are learned only from training rows.

One engineered feature is retained:

    principal_per_month_proxy = credit_amount / duration

It is a simple monthly principal proxy; it omits interest, fees, income, and repayment schedules. It is not an estimated monthly payment.

The applicant form derives accepted categories and numeric limits from the bundled schema and validates the resulting fields. Sensitive attributes and target labels are rejected as prediction inputs. The what-if view changes financial fields and recomputes the derived proxy using the same fitted pipeline.

The displayed FairCredit risk score is 100 x (1 - model-estimated probability of the bad-credit label). Its Low/Moderate/High bands are illustrative thresholds, not a lender's scoring scale.

## Score limitations

The app reports the fitted model's estimated probability and a corresponding illustrative risk score. It does not provide feature-level explanations. The model probability is uncalibrated and is not a measure of confidence, a causal explanation, or a bank decision.

## Fairness experiments

Fairness values are population-level research results in the saved experiment artifacts. The favorable outcome is predicted good credit (class 0); bad credit (class 1) is positive for classification recall and false-negative rates. Demographic parity compares favorable selection rates, equal opportunity compares favorable true-positive rates, and equalized odds also compares favorable false-positive rates. Metric definitions can conflict. Group estimates with fewer than 30 held-out records are flagged; small counts make rates unstable.

Reweighing and Fairlearn reduction results are experimental comparisons, not guarantees. Group attributes remain separate from applicant predictions, and group-specific thresholds are not used by applicant-facing features. Removing sensitive inputs cannot remove proxy discrimination through correlated financial attributes.

## Assessment report and privacy

The report is generated on demand in memory. It includes a report identifier, UTC generation time, risk score and band, bad-label probability, predicted historical label, model description, and fixed-holdout performance metrics with definitions. It does not include the submitted financial profile, a name, contact information, or demographic fields. FairCredit does not persist the report or applicant inputs; a downloaded file is saved by the user's browser. The report also explains the score thresholds, validation limitations, and prototype disclaimer, without factor-level explanations.

## Local setup and tests

Use Python 3.10-3.12 and the pinned requirements.txt. Launch with streamlit run app.py. Install requirements-dev.txt and run python -m pytest tests to exercise data loading, preprocessing, model fit/prediction, fairness, scenario changes, and PDF generation.
