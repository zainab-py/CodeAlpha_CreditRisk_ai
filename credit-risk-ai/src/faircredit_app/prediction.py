"""Input validation and hypothetical financial scenario helpers."""
from __future__ import annotations

import pandas as pd

from faircredit import MODEL_FEATURES, SENSITIVE_ATTRIBUTES, TARGET


def validate_profile(profile: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if len(profile) != 1:
        raise ValueError("Enter exactly one applicant profile.")
    if any(field in profile.columns for field in (*SENSITIVE_ATTRIBUTES, TARGET)):
        raise ValueError("Demographic attributes and outcome labels are not accepted as prediction inputs.")
    missing = [feature for feature in MODEL_FEATURES if feature not in profile.columns]
    if missing:
        raise ValueError("Missing required financial fields: " + ", ".join(missing))
    result = profile[MODEL_FEATURES].copy()
    for column in MODEL_FEATURES:
        if result[column].isna().any():
            raise ValueError(f"{column} is required.")
        if column in reference and reference[column].dtype.kind in "iuf":
            try:
                result[column] = pd.to_numeric(result[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must be numeric.") from exc
            observed = pd.to_numeric(reference[column], errors="coerce").dropna()
            if len(observed) and not observed.min() <= result[column].iloc[0] <= observed.max() * 2:
                raise ValueError(f"{column} is outside the supported input range.")
        else:
            valid = set(reference[column].dropna().astype(str)) if column in reference else set()
            if str(result[column].iloc[0]) not in valid:
                raise ValueError(f"Unsupported {column} category: {result[column].iloc[0]}.")
    return result


def with_engineered_features(profile: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()
    if "principal_per_month_proxy" not in result and {"credit_amount", "duration"}.issubset(result.columns):
        duration = pd.to_numeric(result["duration"], errors="coerce").where(lambda values: values > 0)
        result["principal_per_month_proxy"] = pd.to_numeric(result["credit_amount"], errors="coerce") / duration
    missing = set(MODEL_FEATURES) - set(result.columns)
    if missing:
        raise ValueError("Missing required financial fields: " + ", ".join(sorted(missing)))
    return result[MODEL_FEATURES]


def bad_probability(model, profile: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not provide probability estimates.")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The model is missing the expected bad-credit label.")
    return float(model.predict_proba(profile)[0, classes.index(1)])


def probability_category(probability: float) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between zero and one.")
    if probability < 1 / 3:
        return "Lower estimated bad-label probability"
    if probability < 2 / 3:
        return "Mid-range estimated bad-label probability"
    return "Higher estimated bad-label probability"
