import pandas as pd
import pytest

from faircredit import (
    DATA_COLUMNS, MODEL_FEATURES, TARGET, age_group, load_german_credit,
    make_model_pipeline, prepare_data, train_primary_model,
)
from faircredit_app.prediction import bad_probability, probability_category, validate_profile, with_engineered_features
from faircredit_app.reporting import DISCLAIMER, generate_assessment_pdf
from faircredit import run_audit


def test_dataset_loading_and_schema(data):
    assert len(data) == 1000
    assert set(DATA_COLUMNS).issubset(data.columns)
    assert set(data[TARGET].unique()) == {0, 1}
    assert data[TARGET].value_counts().to_dict() == {0: 700, 1: 300}


def test_preprocessing_normalizes_labels_and_removes_duplicates(data):
    duplicate = pd.concat([data.iloc[[0]], data.iloc[[0]]], ignore_index=True)
    # Recreate source labels, as prepare_data expects all source columns.
    duplicate.loc[:, TARGET] = duplicate[TARGET].map({0: 1, 1: 2})
    prepared = prepare_data(duplicate)
    assert len(prepared) == 1
    assert prepared[TARGET].iloc[0] in (0, 1)


def test_preprocessing_rejects_missing_schema(data):
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_data(data.drop(columns=["purpose"]))


def test_feature_engineering_formula():
    frame = pd.DataFrame({"credit_amount": [1200], "duration": [12]})
    result = with_engineered_features(frame.assign(**{key: "A14" for key in MODEL_FEATURES if key not in frame and key != "principal_per_month_proxy"}))
    assert result.loc[0, "principal_per_month_proxy"] == 100


def test_model_training_excludes_sensitive_inputs(training):
    assert not {"age", "personal_status", "foreign_worker"}.intersection(training.model.feature_names_in_)
    assert training.train_rows == 800
    assert training.test_rows == 200


def test_prediction_and_probability_validation(training, data):
    raw = data.iloc[[0]][MODEL_FEATURES].copy()
    raw.loc[:, "principal_per_month_proxy"] = raw.credit_amount / raw.duration
    p = bad_probability(training.model, raw)
    assert 0 <= p <= 1
    assert probability_category(p)
    with pytest.raises(ValueError, match="between zero and one"):
        probability_category(1.2)


def test_profile_validation_rejects_missing_or_unknown_fields(data):
    with pytest.raises(ValueError, match="Missing required"):
        validate_profile(pd.DataFrame([{"duration": 12}]), data)
    profile = data.iloc[[0]][MODEL_FEATURES].copy()
    profile.loc[:, "checking_status"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_profile(profile, data)
    sensitive = data.iloc[[0]][MODEL_FEATURES + ["age"]].copy()
    with pytest.raises(ValueError, match="Demographic attributes"):
        validate_profile(sensitive, data)


def test_fairness_metrics_and_small_group(data):
    audit = run_audit(data, "foreign_worker")
    assert {"Applicants", "Predicted high risk", "True positive rate", "False positive rate"}.issubset(audit.group_metrics.columns)
    assert audit.group_metrics["Applicants"].sum() == 200
    assert audit.group_metrics.loc[audit.group_metrics["Applicants"] < 30, "small_group_warning"].all()
    assert audit.metrics["Model"].tolist() == ["Unmitigated audit RF", "Reweighed audit RF"]


def test_what_if_updates_engineered_feature(data, training):
    profile = data.iloc[[0]][MODEL_FEATURES].copy()
    profile.loc[:, "credit_amount"] = 1000
    profile.loc[:, "duration"] = 10
    profile.loc[:, "principal_per_month_proxy"] = 100
    scenario = profile.copy()
    scenario.loc[:, "credit_amount"] = 2000
    scenario.loc[:, "employment"] = "A75"
    scenario.loc[:, "principal_per_month_proxy"] = scenario.credit_amount / scenario.duration
    assert scenario.loc[0, "principal_per_month_proxy"] == 200
    assert 0 <= bad_probability(training.model, scenario) <= 1


def test_pdf_generation(training, data):
    report_data = {
        "report_id": "ABC123",
        "bad_probability": .42,
        "risk_score": 58,
        "risk_category": "Moderate risk",
        "predicted_label": "Good-credit history (dataset label 0)",
        "dataset_rows": len(data),
        "training_rows": training.train_rows,
        "test_rows": training.test_rows,
        "holdout_metrics": training.metrics,
        "model_parameters": training.selected_parameters,
    }
    pdf = generate_assessment_pdf(**report_data)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 4000
    assert DISCLAIMER.startswith("Educational and research")
    with pytest.raises(ValueError, match="between zero and one"):
        generate_assessment_pdf(**{**report_data, "bad_probability": 1.2})
    with pytest.raises(ValueError, match="must match"):
        generate_assessment_pdf(**{**report_data, "risk_score": 57})
    with pytest.raises(ValueError, match="risk-score band"):
        generate_assessment_pdf(**{**report_data, "risk_category": "Low risk"})
    with pytest.raises(ValueError, match="add up"):
        generate_assessment_pdf(**{**report_data, "test_rows": 199})
