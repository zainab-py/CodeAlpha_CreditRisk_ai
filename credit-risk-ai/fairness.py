"""Fairlearn group metrics for the German Credit research experiments."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SMALL_GROUP_N = 30


def evaluate_group_fairness(
    y_true_bad: pd.Series | np.ndarray,
    y_pred_bad: pd.Series | np.ndarray,
    sensitive_features: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Return Fairlearn MetricFrames and parity metrics using good credit as favorable.

    The classifier's positive label is bad credit (1). Fairness metrics that
    describe a favorable selection instead invert labels so predicted good
    credit is the favorable outcome. The per-group error metrics retain bad
    credit as positive, which makes false-negative rates easy to interpret.
    """
    y_bad = np.asarray(y_true_bad, dtype=int)
    pred_bad = np.asarray(y_pred_bad, dtype=int)
    groups = pd.Series(np.asarray(sensitive_features), name="audit_group")
    bad_metrics = {
        "sample_count": lambda y_true, y_pred: len(y_true),
        "accuracy": accuracy_score,
        "precision_bad": partial(precision_score, pos_label=1, zero_division=0),
        "recall_bad_TPR": partial(recall_score, pos_label=1, zero_division=0),
        "F1_bad": partial(f1_score, pos_label=1, zero_division=0),
        "false_positive_rate_bad": false_positive_rate,
        "false_negative_rate_bad": false_negative_rate,
    }
    bad_frame = MetricFrame(
        metrics=bad_metrics, y_true=y_bad, y_pred=pred_bad,
        sensitive_features=groups,
    )

    y_good, pred_good = 1 - y_bad, 1 - pred_bad
    favorable_frame = MetricFrame(
        metrics={
            "favorable_selection_rate_good": selection_rate,
            "true_positive_rate_good": true_positive_rate,
            "false_positive_rate_good": false_positive_rate,
        },
        y_true=y_good, y_pred=pred_good, sensitive_features=groups,
    )
    differences = favorable_frame.difference(method="between_groups")
    dp_ratio = demographic_parity_ratio(
        y_true=y_good, y_pred=pred_good, sensitive_features=groups,
        method="between_groups",
    )
    eo_diff = float(differences["true_positive_rate_good"])
    eodds_diff = equalized_odds_difference(
        y_true=y_good, y_pred=pred_good, sensitive_features=groups,
        method="between_groups",
    )
    summary = {
        "demographic_parity_difference": float(differences["favorable_selection_rate_good"]),
        "demographic_parity_ratio": float(dp_ratio),
        "equal_opportunity_difference": eo_diff,
        "equalized_odds_difference": float(eodds_diff),
        "equalized_odds_tpr_good_difference": eo_diff,
        "equalized_odds_fpr_good_difference": float(differences["false_positive_rate_good"]),
    }

    by_group = bad_frame.by_group.reset_index()
    group_name = by_group.columns[0]
    by_group = by_group.rename(columns={
        group_name: "Group", "sample_count": "Applicants", "accuracy": "Accuracy",
        "precision_bad": "Precision (bad positive)", "recall_bad_TPR": "True positive rate (bad)",
        "F1_bad": "F1 (bad positive)", "false_positive_rate_bad": "False positive rate (bad)",
        "false_negative_rate_bad": "False negative rate (bad)",
    })
    by_group["small_group_warning"] = by_group["Applicants"] < SMALL_GROUP_N
    return {
        "summary": summary,
        "by_group": by_group,
        "favorable_by_group": favorable_frame.by_group.reset_index().rename(
            columns={favorable_frame.by_group.index.name or "audit_group": "Group"}
        ),
        "small_group_threshold": SMALL_GROUP_N,
    }
