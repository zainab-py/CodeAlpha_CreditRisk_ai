# Model and fairness experiments

## Reproduce

From the repository root, install `credit-risk-ai/requirements.txt`, then run:

```powershell
python credit-risk-ai/experiments.py
python credit-risk-ai/experiments.py --group personal_status --output credit-risk-ai/reports/experiments/personal_status
python credit-risk-ai/experiments.py --group foreign_worker --output credit-risk-ai/reports/experiments/foreign_worker
```

The runs recorded below used the bundled UCI data, seed 42, an 80/20 stratified split (800 training, 200 holdout), and five-fold shuffled stratified CV on the training partition. Hyperparameters were selected by CV ROC-AUC. Imputation, scaling, and one-hot encoding are inside the pipeline and fit within each training fold. The final holdout did not select parameters or thresholds.

The fixed hyperparameter grids were:

- Logistic Regression: `C ∈ {0.1, 1, 10}` and `class_weight ∈ {None, balanced}`.
- Random Forest: `n_estimators ∈ {100, 250}`, `max_depth ∈ {None, 8}`, `min_samples_leaf ∈ {1, 3}`, and `class_weight ∈ {None, balanced}`.
- XGBoost: `n_estimators ∈ {100, 200}`, `max_depth ∈ {2, 4}`, `learning_rate ∈ {0.05, 0.1}`, and `subsample ∈ {0.8, 1.0}`.

All models exclude age, personal status, and foreign-worker status from predictors. The audit attribute remains separate for MetricFrame evaluation and, in the mitigation experiment, training-only sample weights. The default comparison audits age bands. Separate actual runs and result files are included for the combined personal-status/sex code and foreign-worker status.

## Holdout comparison: age-band audit

Target class 1 is **bad credit** for precision, recall, F1, confusion matrices, and false-negative rate. For demographic parity and the equal-opportunity/equalized-odds metrics, **predicted good credit (class 0)** is the favorable outcome. DP Ratio is the minimum group favorable-selection rate divided by the maximum. Difference values are maximum minus minimum group rates.

| Model | Accuracy | F1 | ROC-AUC | DP Difference | DP Ratio | EO Difference | EOdds Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.770 | 0.511 | 0.794 | 0.139 | 0.844 | 0.071 | 0.071 |
| Random Forest | 0.790 | 0.661 | 0.801 | 0.142 | 0.812 | 0.056 | 0.150 |
| XGBoost | 0.805 | 0.636 | 0.795 | 0.079 | 0.899 | 0.059 | 0.075 |

These are actual results from the fixed 200-row holdout at probability threshold 0.5. XGBoost has the highest accuracy in this run; Random Forest has the highest F1 and recall. The group metrics do not identify an accuracy-only winner.

## Mitigation comparison: age-band audit

Reweighing uses `w(A,Y) = P(A) P(Y) / P(A,Y)` calculated from training rows only. It changes fitting weights, not the applicant model's inputs. All values below are measured on the same holdout at threshold 0.5.

| Model | Accuracy | F1 | ROC-AUC | Brier | DP Difference | EO Difference | EOdds Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, baseline | 0.770 | 0.511 | 0.794 | 0.160 | 0.139 | 0.071 | 0.071 |
| Logistic Regression, reweighed | 0.755 | 0.495 | 0.800 | 0.159 | 0.072 | 0.044 | 0.150 |
| Random Forest, baseline | 0.790 | 0.661 | 0.801 | 0.173 | 0.142 | 0.056 | 0.150 |
| Random Forest, reweighed | 0.785 | 0.645 | 0.797 | 0.174 | 0.052 | 0.090 | 0.169 |
| XGBoost, baseline | 0.805 | 0.636 | 0.795 | 0.154 | 0.079 | 0.059 | 0.075 |
| XGBoost, reweighed | 0.775 | 0.571 | 0.800 | 0.153 | 0.034 | 0.031 | 0.225 |

The changes are mixed: reweighing lowers DP Difference for all three models, but raises EOdds Difference for all three. It also reduces XGBoost recall from 0.567 to 0.500. Brier score changes are small and not uniform. No mitigation is uniformly better across metrics.

### Fairlearn reductions

Experiment C wraps the CV-tuned Logistic Regression pipeline in Fairlearn `ExponentiatedGradient` with the `EqualizedOdds(difference_bound=0.05)` constraint, `eps=0.01`, and 30 iterations. On the age-band holdout, accuracy was 0.755, F1 0.473, DP Difference 0.161, EO Difference 0.071, and EOdds Difference 0.150. It did not meet the requested bound on this empirical holdout. The algorithm returns a randomized mixture and does not expose `predict_proba`, so ROC-AUC, Brier score, and calibration are unavailable for this result. The reduction is exploratory, not a guarantee that a finite-sample model meets a constraint.

## Threshold and recall analysis

The target is bad-credit label 1. A false negative therefore means a source-labeled bad-credit row predicted as good; false-negative rate is `1 - recall_bad`. At threshold 0.5 on the age-band holdout, false-negative rates were 0.600 (Logistic Regression), 0.317 (Random Forest), and 0.433 (XGBoost).

The runner selects a single global F1 threshold from training-only out-of-fold probabilities, then reports its holdout result. The selected thresholds were 0.20, 0.40, and 0.25 respectively. They raised bad-label recall on this holdout to 0.850, 0.817, and 0.767, while accuracy fell to 0.570, 0.685, and 0.710. These are illustrative changes; the intended application's error costs determine the threshold and no threshold is universally optimal.

It also evaluates a predeclared global threshold sensitivity grid (0.30, 0.40, 0.50, 0.60, 0.70) on the holdout for group-metric comparison. Those rows are sensitivity analysis only; none is selected from holdout results. The applicant-facing UI does not use protected attributes to set a decision threshold.

## Group metrics and sample sizes

Fairlearn `MetricFrame` stores group sample counts, accuracy, precision/recall/F1 for bad credit, true/false-positive rates, and false-negative rates. A group below 30 holdout rows is flagged. Age-band holdout counts were 44 (18–25), 111 (26–45), and 45 (46+), so none crossed the warning threshold in this split.

Personal status combines sex and marital status; it is not a standalone sex variable. Its holdout counts were A91=9, A92=61, A93=118, A94=12. Foreign-worker status counts were A201=194 and A202=6. The A91, A94, and A202 subgroup rates are flagged as small and should not support strong claims. For example, Logistic Regression's foreign-worker EOdds Difference was 0.610 in this split, driven by the six-row A202 group; that estimate is highly unstable.

Parity metrics answer different questions and can conflict. DP ignores the true outcome; EO compares favorable true-positive rates; EOdds also includes favorable false-positive rates. The German Credit label is historical, subgroup sizes differ, and one holdout has sampling uncertainty. A small gap on one metric does not establish absence of bias.

## Model selection

The runner marks the training-only out-of-fold Pareto set across ROC-AUC, F1, DP Difference, EO Difference, and EOdds Difference. All three baseline models are non-dominated in the recorded CV comparison, so there is no single model that wins across predictive and fairness objectives. The Streamlit applicant explorer keeps Random Forest as its educational default for continuity with the original project, using its CV-selected age-band parameters (`n_estimators=100`, `max_depth=8`, `min_samples_leaf=3`, balanced class weights). This is not a claim that Random Forest is fair or optimal for real lending.

## Generated artifacts

For each audit attribute, the runner stores:

- `model_comparison.csv` — consolidated baseline, reweighing, threshold, and reduction results.
- `experiment_results.json` — hyperparameters, holdout metrics, classification reports, confusion matrices, group metrics/sample sizes, calibration/ROC/PR curve points, and caveats.
- `model_curves_and_fairness.png`, `confusion_matrices.png`, `confusion_and_group_rates.png`, and `threshold_sensitivity.png`.

The default age-band artifacts are in `reports/experiments/`; other group runs are in their named subfolders. These are one seeded split and not confidence intervals or evidence of performance on current applicants.
