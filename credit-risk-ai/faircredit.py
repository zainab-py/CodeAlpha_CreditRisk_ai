"""Leakage-safe German Credit preprocessing and educational fairness audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path(__file__).resolve().parent / "data" / "german.data"
DATA_SOURCE = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
DATA_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk",
]
NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "existing_credits", "num_dependents", "principal_per_month_proxy",
]
SENSITIVE_ATTRIBUTES = ["age", "personal_status", "foreign_worker"]
TARGET = "credit_risk"
MODEL_FEATURES = [
    column for column in DATA_COLUMNS
    if column not in (*SENSITIVE_ATTRIBUTES, TARGET)
]
MODEL_FEATURES.append("principal_per_month_proxy")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate source fields, normalize the outcome, deduplicate, and engineer features.

    Feature imputers and encoders are intentionally not fitted here. They live
    in the sklearn pipeline and are learned from training rows only.
    """
    missing_columns = [column for column in DATA_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    data = df[DATA_COLUMNS].copy()
    for column in NUMERIC_FEATURES[:-1] + ["age"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    target = pd.to_numeric(data[TARGET], errors="coerce")
    observed_target = set(target.dropna().unique())
    if observed_target and observed_target <= {1, 2} and 2 in observed_target:
        # Source convention: 1=good, 2=bad. Normalize so bad is positive class 1.
        data[TARGET] = target.map({1: 0, 2: 1})
    else:
        data[TARGET] = target
    data = data.dropna(subset=[TARGET])
    if not set(data[TARGET].unique()) <= {0, 1}:
        raise ValueError("credit_risk must be source labels 1=good, 2=bad or normalized labels 0=good, 1=bad.")
    data[TARGET] = data[TARGET].astype(int)

    categorical = [column for column in DATA_COLUMNS if column not in NUMERIC_FEATURES[:-1] + ["age", TARGET]]
    for column in categorical:
        data[column] = (
            data[column].astype("string").str.strip().replace("", np.nan).astype(object)
        )

    duplicate_count = int(data.duplicated(subset=DATA_COLUMNS).sum())
    data = data.drop_duplicates(subset=DATA_COLUMNS, keep="first").copy()
    data.attrs["duplicate_rows_removed"] = duplicate_count
    data.attrs["rows_without_target_removed"] = int(len(df) - data.shape[0] - duplicate_count)

    # Average principal amount per loan month. This is not a payment estimate:
    # the dataset does not provide the interest schedule or borrower income.
    duration = data["duration"].where(data["duration"] > 0)
    data["principal_per_month_proxy"] = data["credit_amount"] / duration
    return data


def load_german_credit() -> pd.DataFrame:
    """Load the checked-in UCI source-format data, requiring no network access."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Bundled German Credit data not found: {DATA_FILE}")
    raw = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=DATA_COLUMNS)
    return prepare_data(raw)


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 25, 45, np.inf], labels=["18-25", "26-45", "46+"])


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build imputation/scaling/encoding transforms to be fitted within a Pipeline."""
    numeric = [column for column in x.columns if column in NUMERIC_FEATURES]
    categorical = [column for column in x.columns if column not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def make_model_pipeline(x: pd.DataFrame, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(x)),
        ("classifier", RandomForestClassifier(
            # Single-process fitting is reproducible and works in restricted
            # Windows environments where joblib workers cannot create pipes.
            random_state=seed, class_weight="balanced", n_jobs=1,
        )),
    ])


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Return train/test indices using the target's stratified distribution."""
    return train_test_split(
        df.index.to_numpy(), test_size=test_size, random_state=seed,
        stratify=df[TARGET],
    )


def model_metrics(y_true: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probability)),
    }


@dataclass
class TrainingBundle:
    model: Pipeline
    metrics: dict[str, float]
    train_rows: int
    test_rows: int
    selected_parameters: dict[str, Any]


def train_primary_model(df: pd.DataFrame) -> TrainingBundle:
    """Fit the applicant model without age, personal status, or foreign-worker status."""
    data = prepare_data(df)
    train_idx, test_idx = split_data(data)
    x_train = data.loc[train_idx, MODEL_FEATURES]
    x_test = data.loc[test_idx, MODEL_FEATURES]
    y_train, y_test = data.loc[train_idx, TARGET], data.loc[test_idx, TARGET]
    model = make_model_pipeline(x_train)
    # Five-fold training-only CV in experiments.py selected this RF setting
    # for the fixed seed and age-band audit. It is a useful app default, not a
    # uniquely dominant or universally recommended model.
    selected_parameters = {
        "n_estimators": 100, "max_depth": 8, "min_samples_leaf": 3,
        "class_weight": "balanced",
    }
    model.set_params(**{f"classifier__{key}": value for key, value in selected_parameters.items()})
    # Imputation, scaling, and category discovery are fitted on training rows only.
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    return TrainingBundle(model, model_metrics(y_test, prediction, probability), len(train_idx), len(test_idx), selected_parameters)


def group_metrics(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame({
        "actual": np.asarray(y_true), "predicted": np.asarray(y_pred), "group": np.asarray(groups),
    })
    for group, part in frame.dropna(subset=["group"]).groupby("group", observed=True):
        actual, predicted = part.actual.to_numpy(), part.predicted.to_numpy()
        positives, negatives = actual == 1, actual == 0
        rows.append({
            "Group": str(group), "Applicants": len(part),
            "Predicted high risk": float(predicted.mean()),
            "True positive rate": float(predicted[positives].mean()) if positives.any() else np.nan,
            "False positive rate": float(predicted[negatives].mean()) if negatives.any() else np.nan,
            "False negative rate": float(1 - predicted[positives].mean()) if positives.any() else np.nan,
            "Accuracy": float(accuracy_score(actual, predicted)),
            "Precision": float(precision_score(actual, predicted, zero_division=0)),
            "Recall": float(recall_score(actual, predicted, zero_division=0)),
            "F1": float(f1_score(actual, predicted, zero_division=0)),
            "small_group_warning": len(part) < 30,
        })
    return pd.DataFrame(rows)


def fairness_summary(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {"selection_gap": np.nan, "tpr_gap": np.nan, "fpr_gap": np.nan, "selection_ratio": np.nan}

    def gap(column: str) -> float:
        values = metrics[column].dropna()
        return float(values.max() - values.min()) if len(values) else np.nan

    rates = metrics["Predicted high risk"].dropna()
    ratio = float(rates.min() / rates.max()) if len(rates) and rates.max() else np.nan
    return {"selection_gap": gap("Predicted high risk"), "tpr_gap": gap("True positive rate"),
            "fpr_gap": gap("False positive rate"), "selection_ratio": ratio}


@dataclass
class AuditResult:
    metrics: pd.DataFrame
    group_metrics: pd.DataFrame
    mitigated_group_metrics: pd.DataFrame
    note: str


def run_audit(df: pd.DataFrame, group_attribute: str) -> AuditResult:
    """Compare an unmitigated and reweighed model; all protected inputs stay audit-only."""
    data = prepare_data(df)
    if group_attribute not in data.columns:
        raise ValueError(f"Audit attribute not present: {group_attribute}")
    groups = (
        pd.Series(age_group(data["age"]), index=data.index)
        if group_attribute == "age_group" else data[group_attribute]
    )
    train_idx, test_idx = split_data(data)
    x_train = data.loc[train_idx, MODEL_FEATURES]
    x_test = data.loc[test_idx, MODEL_FEATURES]
    y_train, y_test = data.loc[train_idx, TARGET], data.loc[test_idx, TARGET]
    group_train, group_test = groups.loc[train_idx].astype("string"), groups.loc[test_idx].astype("string")

    baseline = make_model_pipeline(x_train)
    baseline.fit(x_train, y_train)
    prediction = baseline.predict(x_test)
    probability = baseline.predict_proba(x_test)[:, list(baseline.classes_).index(1)]

    # Reweigh each group/outcome cell to the product of its marginal rates.
    cells = pd.DataFrame({"group": group_train, "target": y_train})
    group_rates = cells["group"].value_counts(normalize=True)
    target_rates = cells["target"].value_counts(normalize=True)
    joint_rates = cells.groupby(["group", "target"], observed=True).size() / len(cells)
    weights = np.asarray([
        (group_rates.loc[group] * target_rates.loc[target]) / joint_rates.loc[(group, target)]
        for group, target in zip(group_train, y_train)
    ])
    reweighed = make_model_pipeline(x_train)
    reweighed.fit(x_train, y_train, classifier__sample_weight=weights)
    prediction_reweighed = reweighed.predict(x_test)
    probability_reweighed = reweighed.predict_proba(x_test)[:, list(reweighed.classes_).index(1)]

    metrics = pd.DataFrame([
        {"Model": "Unmitigated audit RF", **model_metrics(y_test, prediction, probability)},
        {"Model": "Reweighed audit RF", **model_metrics(y_test, prediction_reweighed, probability_reweighed)},
    ])
    note = (
        "Both audit models use the same stratified 80/20 split and exclude age, personal status, and foreign-worker status "
        "from predictors. The selected demographic field is retained only to calculate group metrics and training weights. "
        "This experiment does not establish that the applicant model is fair."
    )
    return AuditResult(metrics, group_metrics(y_test, prediction, group_test),
                       group_metrics(y_test, prediction_reweighed, group_test), note)
