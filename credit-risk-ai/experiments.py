"""Reproducible predictive/fairness experiment runner for FairCredit.

Run from the repository root with ``python credit-risk-ai/experiments.py``.
All tuning, mitigation-weight calculation, and threshold selection use the
training split only. The final 20% holdout is used for evaluation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

from faircredit import (
    DATA_FILE, MODEL_FEATURES, TARGET, age_group, load_german_credit,
    make_model_pipeline, model_metrics, prepare_data,
)
from fairness import evaluate_group_fairness


SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
FIXED_THRESHOLD_SENSITIVITY = [0.30, 0.40, 0.50, 0.60, 0.70]
REPORT_DIR = Path(__file__).resolve().parent / "reports" / "experiments"


def get_audit_groups(data: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute == "age_group":
        return pd.Series(age_group(data["age"]), index=data.index, name=attribute)
    if attribute not in data.columns:
        raise ValueError(f"Unknown audit field: {attribute}")
    return data[attribute].astype("string").rename(attribute)


def build_searches(x_train: pd.DataFrame) -> dict[str, tuple[GridSearchCV, dict[str, Any]]]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "recall_bad": "recall"}
    specifications = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED, solver="liblinear"),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"classifier__n_estimators": [100, 250], "classifier__max_depth": [None, 8],
             "classifier__min_samples_leaf": [1, 3], "classifier__class_weight": [None, "balanced"]},
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=SEED, n_jobs=1, eval_metric="logloss",
                objective="binary:logistic", tree_method="hist",
            ),
            {"classifier__n_estimators": [100, 200], "classifier__max_depth": [2, 4],
             "classifier__learning_rate": [0.05, 0.1], "classifier__subsample": [0.8, 1.0]},
        ),
    }
    searches = {}
    for name, (classifier, params) in specifications.items():
        pipeline = make_model_pipeline(x_train)
        pipeline.set_params(classifier=classifier)
        # All transformation steps remain inside the estimator and are fitted
        # independently inside each cross-validation fold.
        search = GridSearchCV(
            estimator=pipeline, param_grid=params, scoring=scoring, refit="roc_auc",
            cv=cv, n_jobs=1, return_train_score=False, error_score="raise",
        )
        searches[name] = (search, params)
    return searches


def choose_f1_threshold(y_true: np.ndarray, p_bad: np.ndarray) -> tuple[float, float]:
    candidates = np.arange(0.10, 0.91, 0.05)
    scores = [(float(t), f1_score(y_true, p_bad >= t, zero_division=0)) for t in candidates]
    threshold, score = max(scores, key=lambda pair: (pair[1], pair[0]))
    return threshold, float(score)


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    probability_bad: np.ndarray | None,
    prediction_bad: np.ndarray,
    groups: pd.Series,
    threshold: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(prediction_bad, dtype=int)
    if probability_bad is None:
        metrics = {
            "Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "ROC-AUC": None, "Brier score": None,
            "False negative rate": float(1.0 - recall_score(y, pred, zero_division=0)),
        }
        curves = {"roc": None, "precision_recall": None, "calibration": None}
        roc_auc = None
    else:
        p_bad = np.asarray(probability_bad, dtype=float)
        metrics = model_metrics(y, pred, p_bad)
        metrics["Brier score"] = float(brier_score_loss(y, p_bad))
        metrics["False negative rate"] = float(1.0 - metrics["Recall"])
        fpr, tpr, roc_thresholds = roc_curve(y, p_bad)
        precision, recall, pr_thresholds = precision_recall_curve(y, p_bad)
        fraction_positive, mean_predicted = calibration_curve(y, p_bad, n_bins=5, strategy="quantile")
        curves = {
            "roc": {"false_positive_rate": fpr, "true_positive_rate": tpr, "thresholds": roc_thresholds},
            "precision_recall": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
            "calibration": {"fraction_positive": fraction_positive, "mean_predicted": mean_predicted},
        }
        roc_auc = metrics["ROC-AUC"]
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["Good credit (0)", "Bad credit (1)"],
        output_dict=True, zero_division=0,
    )
    fairness = evaluate_group_fairness(y, pred, groups)
    confusion = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    row = {
        "Model": model_name, "Accuracy": metrics["Accuracy"], "Precision": metrics["Precision"],
        "Recall": metrics["Recall"], "F1": metrics["F1"], "ROC-AUC": roc_auc,
        "Brier score": metrics.get("Brier score"), "False negative rate": metrics.get("False negative rate"),
        "DP Difference": fairness["summary"]["demographic_parity_difference"],
        "DP Ratio": fairness["summary"]["demographic_parity_ratio"],
        "EO Difference": fairness["summary"]["equal_opportunity_difference"],
        "EOdds Difference": fairness["summary"]["equalized_odds_difference"],
        "Decision threshold": threshold,
    }
    detail = {
        "metrics": metrics, "confusion_matrix_labels": ["Good credit (0)", "Bad credit (1)"],
        "confusion_matrix": confusion, "classification_report": report,
        "fairness": fairness, "curves": curves,
    }
    return row, detail


def calculate_reweighing_weights(y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Reweigh group/outcome intersections using training rows only."""
    cells = pd.DataFrame({"group": np.asarray(groups), "target": np.asarray(y, dtype=int)})
    group_rates = cells["group"].value_counts(normalize=True)
    target_rates = cells["target"].value_counts(normalize=True)
    joint_rates = cells.groupby(["group", "target"], observed=True).size() / len(cells)
    return np.asarray([
        (group_rates.loc[group] * target_rates.loc[target]) / joint_rates.loc[(group, target)]
        for group, target in zip(cells["group"], cells["target"])
    ])


def is_pareto_efficient(rows: list[dict[str, Any]]) -> list[bool]:
    """Keep candidates not dominated on train-only OOF performance and fairness."""
    objectives = [
        ("CV OOF ROC-AUC", True), ("CV OOF F1", True),
        ("CV OOF DP Difference", False), ("CV OOF EO Difference", False),
        ("CV OOF EOdds Difference", False),
    ]
    keep = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            comparisons = []
            strict = []
            for key, higher_is_better in objectives:
                a, b = row.get(key), other.get(key)
                if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
                    continue
                comparisons.append(b >= a if higher_is_better else b <= a)
                strict.append(b > a if higher_is_better else b < a)
            if comparisons and all(comparisons) and any(strict):
                dominated = True
                break
        keep.append(not dominated)
    return keep


def to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return to_jsonable(value.item())
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (dict, list, tuple, np.ndarray)) and pd.isna(value):
        return None
    return value


def plot_results(
    baseline_details: dict[str, dict[str, Any]],
    reweighed_details: dict[str, dict[str, Any]],
    group_results: dict[str, dict[str, pd.DataFrame]],
    thresholds: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {"Logistic Regression": "#2878B5", "Random Forest": "#F28E2B", "XGBoost": "#59A14F"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for name, detail in baseline_details.items():
        color = colors[name]
        roc = detail["curves"]["roc"]
        pr = detail["curves"]["precision_recall"]
        cal = detail["curves"]["calibration"]
        axes[0, 0].plot(roc["false_positive_rate"], roc["true_positive_rate"], label=name, color=color)
        axes[0, 1].plot(pr["recall"], pr["precision"], label=name, color=color)
        axes[1, 0].plot(cal["mean_predicted"], cal["fraction_positive"], marker="o", label=f"{name} baseline", color=color)
        reweighted_cal = reweighed_details[name]["curves"]["calibration"]
        axes[1, 0].plot(reweighted_cal["mean_predicted"], reweighted_cal["fraction_positive"],
                        marker="x", linestyle="--", label=f"{name} reweighed", color=color, alpha=.75)
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="#888")
    axes[0, 0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[0, 1].set(title="Precision–recall curves", xlabel="Recall (bad-credit class)", ylabel="Precision")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="#888")
    axes[1, 0].set(title="Calibration curves", xlabel="Mean predicted bad-credit probability", ylabel="Observed bad-credit frequency")
    for ax in axes.flat[:3]:
        ax.legend()
        ax.grid(alpha=.2)

    ax = axes[1, 1]
    baseline_names = list(baseline_details)
    width = .34
    x = np.arange(len(baseline_names))
    base_dp, mitigated_dp = [], []
    for name in baseline_names:
        base = evaluate_group_fairness(
            group_results[name]["y_true"], group_results[name]["baseline_prediction"],
            group_results[name]["groups"],
        )["summary"]["demographic_parity_difference"]
        mitigated = evaluate_group_fairness(
            group_results[name]["y_true"], group_results[name]["reweighed_prediction"],
            group_results[name]["groups"],
        )["summary"]["demographic_parity_difference"]
        base_dp.append(base)
        mitigated_dp.append(mitigated)
    ax.bar(x-width/2, base_dp, width, label="Baseline", color="#777")
    ax.bar(x+width/2, mitigated_dp, width, label="Reweighed", color="#8CD17D")
    ax.set_xticks(x, baseline_names, rotation=15)
    ax.set(title="Demographic parity gap", ylabel="Difference in favorable (good-credit) rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_curves_and_fairness.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(baseline_details), figsize=(5 * len(baseline_details), 4.5))
    if len(baseline_details) == 1:
        axes = [axes]
    for ax, (name, detail) in zip(axes, baseline_details.items()):
        cm = np.asarray(detail["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Good", "Bad"])
        ax.set_yticks([0, 1], ["Good", "Bad"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title(name)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.suptitle("Holdout confusion matrices (class 1 = bad credit)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["True positive rate (bad)", "False positive rate (bad)"]):
        for name, frames in group_results.items():
            before, after = frames["baseline_groups"], frames["reweighed_groups"]
            ax.plot(before["Group"], before[metric], marker="o", label=f"{name} baseline")
            ax.plot(after["Group"], after[metric], marker="x", linestyle="--", label=f"{name} reweighed")
        ax.set_title(f"{metric} by audit group")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_and_group_rates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, frame in thresholds.groupby("Base model"):
        ax.plot(frame["Threshold"], frame["DP Difference"], marker="o", label=f"{name} DP")
        ax.plot(frame["Threshold"], frame["EO Difference"], marker="x", linestyle="--", label=f"{name} EO")
    ax.set(title="Predeclared global threshold sensitivity (holdout)", xlabel="Global probability threshold for bad-credit label", ylabel="Group difference")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sensitivity.png", dpi=160)
    plt.close(fig)


def run_experiments(group_attribute: str = "age_group", output_dir: Path = REPORT_DIR) -> dict[str, Any]:
    data = prepare_data(load_german_credit())
    x = data[MODEL_FEATURES]
    y = data[TARGET]
    groups = get_audit_groups(data, group_attribute)
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    searches = build_searches(x_train)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    baseline_details: dict[str, dict[str, Any]] = {}
    group_results: dict[str, dict[str, Any]] = {}
    chosen_thresholds: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    baseline_cv_rows: list[dict[str, Any]] = []

    for model_name, (search, _) in searches.items():
        print(f"Tuning {model_name} ({CV_FOLDS}-fold stratified CV)…", flush=True)
        search.fit(x_train, y_train)
        best = search.best_estimator_
        # OOF predictions on training rows are used to assess candidate models
        # and choose the F1 threshold. The test split is never used for tuning.
        oof_probability = cross_val_predict(
            clone(best), x_train, y_train, cv=cv, method="predict_proba", n_jobs=1,
        )[:, 1]
        oof_prediction = (oof_probability >= 0.5).astype(int)
        oof_fairness = evaluate_group_fairness(y_train, oof_prediction, groups_train)
        threshold, threshold_oof_f1 = choose_f1_threshold(np.asarray(y_train), oof_probability)
        chosen_thresholds[model_name] = threshold
        best.fit(x_train, y_train)
        p_test = best.predict_proba(x_test)[:, list(best.classes_).index(1)]
        prediction = (p_test >= 0.5).astype(int)
        row, detail = evaluate_predictions(model_name, y_test, p_test, prediction, groups_test, 0.5)
        row.update({
            "Best parameters": search.best_params_, "CV best ROC-AUC": float(search.best_score_),
            "CV OOF ROC-AUC": float(roc_auc_score(y_train, oof_probability)),
            "CV OOF F1": float(f1_score(y_train, oof_prediction, zero_division=0)),
            "CV OOF DP Difference": oof_fairness["summary"]["demographic_parity_difference"],
            "CV OOF EO Difference": oof_fairness["summary"]["equal_opportunity_difference"],
            "CV OOF EOdds Difference": oof_fairness["summary"]["equalized_odds_difference"],
            "OOF F1 threshold (training only)": threshold,
            "OOF F1 at chosen threshold": threshold_oof_f1,
        })
        rows.append(row)
        baseline_cv_rows.append(row)
        detail.update({
            "best_parameters": search.best_params_, "cv_best_roc_auc": float(search.best_score_),
            "cv_oof": {"metrics_at_0_5": model_metrics(y_train, oof_prediction, oof_probability),
                       "fairness": oof_fairness, "f1_threshold": threshold,
                       "f1_at_threshold": threshold_oof_f1},
            "test_group_metrics": detail["fairness"]["by_group"],
        })
        details[model_name] = detail
        baseline_details[model_name] = detail
        fitted[model_name] = best

        # Reweighing weights use only the training partition and its sensitive
        # group labels. No test-row group or target enters the weights.
        weights = calculate_reweighing_weights(y_train, groups_train)
        reweighed_model = clone(best)
        reweighed_model.fit(x_train, y_train, classifier__sample_weight=weights)
        p_reweighed = reweighed_model.predict_proba(x_test)[:, list(reweighed_model.classes_).index(1)]
        pred_reweighed = (p_reweighed >= 0.5).astype(int)
        reweighed_row, reweighed_detail = evaluate_predictions(
            f"{model_name} (reweighed)", y_test, p_reweighed, pred_reweighed, groups_test, 0.5,
        )
        reweighed_row["Mitigation"] = "Training reweighing"
        rows.append(reweighed_row)
        details[reweighed_row["Model"]] = reweighed_detail
        group_results[model_name] = {
            "y_true": np.asarray(y_test), "groups": np.asarray(groups_test),
            "baseline_prediction": prediction, "reweighed_prediction": pred_reweighed,
            "baseline_groups": detail["fairness"]["by_group"],
            "reweighed_groups": reweighed_detail["fairness"]["by_group"],
        }

        # Threshold grid is fixed before evaluating the holdout. It is a
        # sensitivity analysis only: no threshold is selected from test data.
        for fixed_threshold in FIXED_THRESHOLD_SENSITIVITY:
            fixed_prediction = (p_test >= fixed_threshold).astype(int)
            threshold_row, threshold_detail = evaluate_predictions(
                f"{model_name} @ {fixed_threshold:.2f}", y_test, p_test, fixed_prediction,
                groups_test, fixed_threshold,
            )
            threshold_row["Base model"] = model_name
            threshold_row["Threshold"] = fixed_threshold
            details[threshold_row["Model"]] = threshold_detail
            rows.append(threshold_row)

        # The F1 threshold was selected on OOF training probabilities only.
        threshold_prediction = (p_test >= threshold).astype(int)
        threshold_row, threshold_detail = evaluate_predictions(
            f"{model_name} (OOF F1 threshold)", y_test, p_test, threshold_prediction,
            groups_test, threshold,
        )
        threshold_row["Mitigation"] = "Global threshold, chosen on training OOF"
        rows.append(threshold_row)
        details[threshold_row["Model"]] = threshold_detail

    # Fairlearn ExponentiatedGradient uses the tuned Logistic Regression
    # pipeline and EqualizedOdds(difference_bound=.05); sensitive values are
    # supplied only to the reduction during training and to held-out metrics.
    logistic = fitted["Logistic Regression"]
    reduction = ExponentiatedGradient(
        estimator=clone(logistic), constraints=EqualizedOdds(difference_bound=0.05),
        eps=0.01, max_iter=30, sample_weight_name="classifier__sample_weight",
    )
    reduction.fit(x_train, y_train, sensitive_features=groups_train)
    reduction_prediction = reduction.predict(x_test, random_state=SEED)
    reduction_row, reduction_detail = evaluate_predictions(
        "Logistic Regression (ExponentiatedGradient, EO)", y_test, None,
        reduction_prediction, groups_test, None,
    )
    reduction_row["Mitigation"] = "Fairlearn ExponentiatedGradient, EqualizedOdds ε=0.05"
    reduction_row["Brier score"] = None
    reduction_row["ROC-AUC"] = None
    reduction_detail["note"] = "Reduction prediction is randomized and has no supported predict_proba; ROC-AUC, Brier score, and calibration are not available."
    rows.append(reduction_row)
    details[reduction_row["Model"]] = reduction_detail

    # Determine a train-OOF Pareto set, not a single winner. Accuracy alone is
    # not a model-selection objective, and no holdout values enter this set.
    for row, pareto in zip(baseline_cv_rows, is_pareto_efficient(baseline_cv_rows)):
        row["CV Pareto efficient (performance + fairness)"] = pareto

    # Keep each model's comprehensive required comparison fields in the
    # consolidated output, including mitigation and threshold variants.
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    result = {
        "metadata": {
            "dataset": str(DATA_FILE.name), "records": len(data), "predictors": len(MODEL_FEATURES),
            "target_mapping": "0=good/favorable, 1=bad/adverse", "favorable_prediction": "predicted good credit (class 0)",
            "audit_attribute": group_attribute, "train_rows": len(train_idx), "test_rows": len(test_idx),
            "random_seed": SEED, "cv": f"{CV_FOLDS}-fold StratifiedKFold on training data only",
            "threshold_sensitivity": FIXED_THRESHOLD_SENSITIVITY,
            "threshold_sensitivity_note": "Fixed global thresholds are reported as holdout sensitivity analysis only; no threshold is selected from holdout results.",
            "model_selection_note": "CV OOF Pareto efficiency jointly considers ROC-AUC/F1 and DP/EO/EOdds gaps; no single model is automatically promoted.",
            "group_small_n_warning_threshold": 30,
        },
        "models_and_experiments": details,
        "comparison": rows,
        "limitations": [
            "Good-credit prediction is treated as the favorable outcome for DP/EO/EOdds; bad-credit is positive for classification recall and FNR.",
            "Fairness metrics can conflict, depend on the label and context, and do not prove absence of discrimination.",
            "Subgroups with fewer than 30 held-out rows are flagged; their point estimates are unstable and should not support strong conclusions.",
            "The 1994 German sample, its target labels, and the asymmetric source cost matrix limit generalization.",
            "The applicant model excludes sensitive fields, but correlated predictors may remain proxies.",
            "ExponentiatedGradient produces a randomized mixture and does not expose predict_proba; calibration metrics are unavailable for it.",
        ],
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(to_jsonable(result), indent=2, allow_nan=False), encoding="utf-8",
    )

    # Curve objects for tuned baseline models only.
    test_details = {name: details[name] for name in searches}
    reweighed_details = {name: details[f"{name} (reweighed)"] for name in searches}
    plot_results(test_details, reweighed_details, group_results,
                 summary[summary.get("Threshold", pd.Series(index=summary.index, dtype=float)).notna()], output_dir)
    print(f"Wrote actual results to {output_dir}")
    print(summary[["Model", "Accuracy", "F1", "ROC-AUC", "DP Difference", "EO Difference", "EOdds Difference"]].to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["age_group", "personal_status", "foreign_worker"], default="age_group")
    parser.add_argument("--output", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    run_experiments(args.group, args.output)


if __name__ == "__main__":
    main()
