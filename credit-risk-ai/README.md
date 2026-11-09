# FairCredit app

FairCredit is an educational Streamlit tool for exploring a credit-risk model, group-level prediction differences, and hypothetical changes to financial inputs. It does not provide feature-level explanations.

> **Educational and research use only.** It does not make actual bank credit decisions, guarantee loan approval, provide financial advice, or replace professional financial assessment.

> **Public demo privacy:** use hypothetical examples only. Do not enter real applicant or sensitive financial information; the live app processes submitted inputs on its hosting server.

## Live demo

Open the [FairCredit app](https://codealphacreditriskai-summerinternship.streamlit.app/). The public app is hosted on Streamlit Community Cloud from this repository's `main` branch, using `app.py` and Python 3.12.

## Start

Use Python 3.10–3.12. The preserved original model was serialized with scikit-learn 1.6.1. The app now trains its active applicant model from the bundled data rather than loading that old artifact, because the old feature set included demographic attributes.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app includes Overview, Credit assessment, What-if simulator, Fairness insights, Model insights, and Assessment report. The assessor requests financial fields only. Research views show aggregate results from the bundled public dataset.

Run the test suite after installing development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

Run the reproducible model comparison from the repository root after installing `requirements-experiments.txt`: `python credit-risk-ai/experiments.py`. It produces CSV/JSON results and comparison plots in `reports/experiments/`. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the exact split, CV settings, fairness definitions, and recorded values.

## Dataset and preprocessing

The original UCI `german.data` source is included in `data/german.data`; the app does not need network access. See [`DATASET.md`](DATASET.md) for its measured record/target counts, field descriptions and codes, data quality checks, outlier investigation, sensitive-attribute limitations, and the rationale for the one retained engineered feature.

The applicant model excludes age, personal status/sex code, and foreign-worker status. Those attributes remain in the separate fairness audit pathway. The selected comparison attribute is also withheld from audit-model predictors. Correlated features can still act as proxies.

See [METHODOLOGY.md](METHODOLOGY.md) for applicant flows, score limitations, and report privacy details.

An uploaded audit CSV must include the 20 source predictors and `credit_risk`, with target values `1=good, 2=bad` or `0=good, 1=bad`. Missing feature values are imputed inside the fitted pipeline; duplicate rows are detected and de-duplicated before the stratified split.

## Original baseline

The notebook's Logistic Regression and Random Forest results are preserved in the repository root README and the notebook. The Random Forest artifact remains in the repository as an original artifact, but it is not used for the current applicant score because it uses sensitive demographic predictors.

Fairness comparisons are educational and exploratory. They do not certify fair or lawful outcomes. The source dataset is small and historical; metric gaps and what-if changes are not causal advice. The original analysis notebook retains its separate historical SHAP analysis; the Streamlit app does not load it.
