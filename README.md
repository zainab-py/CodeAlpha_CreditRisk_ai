# FairCredit

An educational Streamlit application for exploring historical credit-risk model outputs, group-level differences, and hypothetical changes to financial attributes. It does not provide feature-level explanations.

> **Educational and research use only.** FairCredit does not make actual bank credit decisions, guarantee loan approval, provide financial advice, or replace professional financial assessment. Model outputs reflect historical data patterns and are not causal explanations.

## Live app

Try the [FairCredit public demo](https://codealphacreditriskai-summerinternship.streamlit.app/). Use hypothetical examples only—do not enter real applicant or sensitive financial information. The hosted app processes submitted inputs on its server.

## Run locally

Use Python 3.10–3.12. The saved model was serialized with scikit-learn 1.6.1; use the pinned versions in `credit-risk-ai/requirements.txt` to avoid pickle incompatibilities.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r credit-risk-ai/requirements.txt
streamlit run credit-risk-ai/app.py
```

The inspected Statlog German Credit sample is bundled for offline use. The applicant experience uses hypothetical financial fields; demographic comparisons are available only as aggregate research views. See the [dataset and preprocessing record](credit-risk-ai/DATASET.md) for source, encodings, data checks, sensitive attributes, and feature engineering.

## Deployment

The live app is hosted on [Streamlit Community Cloud](https://share.streamlit.io/) from the `main` branch, using `credit-risk-ai/app.py` and Python 3.12. Dependencies are in `credit-risk-ai/requirements.txt`; no secrets are required.

See [METHODOLOGY.md](credit-risk-ai/METHODOLOGY.md) for score limitations, what-if behavior, and PDF report details. Tests are in credit-risk-ai/tests; install credit-risk-ai/requirements-dev.txt before running them. The original analysis notebook retains a separate historical SHAP analysis; it is not loaded by the app.

Run the model comparison and fairness experiments after installing `credit-risk-ai/requirements-experiments.txt` with `python credit-risk-ai/experiments.py`; see [EXPERIMENTS.md](credit-risk-ai/EXPERIMENTS.md) for the reproducible protocol and actual recorded results.

## Original recorded results

The figures below are the original notebook's recorded results for one stratified 80/20 split. They are retained unchanged and are not recalculated by the app.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7400 | 0.5465 | 0.7833 | 0.6438 | 0.8092 |
| Random Forest | 0.7800 | 0.7500 | 0.4000 | 0.5217 | 0.7973 |

## Repository map

- `credit-risk-ai/app.py` — Streamlit app.
- `credit-risk-ai/faircredit.py` — data validation, group metrics, audit model, and reweighing.
- `credit-risk-ai/fairness.py` — Fairlearn `MetricFrame` and parity metrics.
- `credit-risk-ai/experiments.py` — CV tuning, held-out evaluation, mitigation experiments, and reports.
- `credit-risk-ai/credit_risk_model.pkl` — preserved original Random Forest artifact; the applicant app uses a newly fitted pipeline without sensitive predictors.
- `credit-risk-ai/data/german.data` — inspected original-format dataset, retained with UCI attribution.
- `credit-risk-ai/DATASET.md` — dataset profile, preprocessing, outliers, and limitations.
- `credit-risk-ai/EXPERIMENTS.md` — modeling/fairness protocol and actual experimental results.
- `credit-risk-ai/requirements.txt` — pinned Streamlit app dependencies.
- `CreditRisk_AI.ipynb` — original analysis and recorded baseline outputs.
- `PROJECT_AUDIT.md` — architecture audit and upgrade notes.
