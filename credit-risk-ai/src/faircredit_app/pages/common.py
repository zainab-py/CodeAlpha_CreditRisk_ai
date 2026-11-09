from __future__ import annotations

import pandas as pd
import streamlit as st

from faircredit import MODEL_FEATURES
from faircredit_app.prediction import with_engineered_features

LABELS = {
    "checking_status": "Checking account", "duration": "Duration (months)",
    "credit_history": "Credit history", "purpose": "Loan purpose",
    "credit_amount": "Loan amount (DM)", "savings_status": "Savings",
    "employment": "Employment duration", "installment_rate": "Installment rate (1-4)",
    "other_debtors": "Other debtor or guarantor", "residence_since": "Years at address",
    "property": "Property", "other_installment_plans": "Other installment plans",
    "housing": "Housing", "existing_credits": "Other credits",
    "job": "Job category", "num_dependents": "Dependents", "telephone": "Telephone",
}


def input_form(data: pd.DataFrame, key: str = "profile") -> pd.DataFrame:
    numeric = {"duration", "credit_amount", "installment_rate", "residence_since", "existing_credits", "num_dependents"}
    values = {}
    sections = [
        ("Loan details", ["credit_amount", "duration", "purpose", "installment_rate"]),
        ("Credit history", ["checking_status", "credit_history", "savings_status", "employment", "existing_credits", "other_installment_plans"]),
        ("Financial and housing details", ["other_debtors", "residence_since", "property", "housing", "job", "num_dependents", "telephone"]),
    ]
    for section, features in sections:
        st.markdown(f'<div class="fc-section">{section}</div>', unsafe_allow_html=True)
        if section == "Loan details":
            st.markdown('<div class="fc-note">Loan amounts use the historical dataset currency unit (DM).</div>', unsafe_allow_html=True)
        for start in range(0, len(features), 2):
            columns = st.columns(2, gap="large")
            for column, feature in zip(columns, features[start:start + 2]):
                with column:
                    if feature in numeric:
                        series = pd.to_numeric(data[feature], errors="coerce").dropna()
                        low, high = int(max(1, series.min())), int(max(2, series.max()))
                        default = min(max(int(series.median()), low), high)
                        values[feature] = st.number_input(
                            LABELS[feature], min_value=low, max_value=high, value=default,
                            step=1, key=f"{key}_{feature}",
                        )
                    else:
                        options = sorted(data[feature].dropna().astype(str).unique().tolist())
                        values[feature] = st.selectbox(
                            LABELS[feature], options,
                            format_func=lambda code: f"{code} - {code_description(code)}",
                            key=f"{key}_{feature}",
                        )
    return with_engineered_features(pd.DataFrame([values]))


def code_description(code: str) -> str:
    descriptions = {
        "A11": "below 0 DM", "A12": "0-200 DM", "A13": "200 DM or more", "A14": "no checking account",
        "A30": "no credit/all paid", "A31": "all paid at this bank", "A32": "paid on time", "A33": "past delays", "A34": "critical/other credits",
        "A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A48": "retraining", "A49": "business", "A410": "other purpose",
        "A61": "under 100 DM", "A62": "100-500 DM", "A63": "500-1,000 DM", "A64": "1,000+ DM", "A65": "unknown/no savings",
        "A71": "unemployed", "A72": "under 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": "7+ years",
        "A101": "none", "A102": "co-applicant", "A103": "guarantor", "A121": "real estate", "A122": "savings/life insurance", "A123": "car/other property", "A124": "unknown/no property",
        "A141": "bank plan", "A142": "store plan", "A143": "none", "A151": "rent", "A152": "own home", "A153": "rent-free",
        "A171": "unskilled, non-resident", "A172": "unskilled resident", "A173": "skilled employee/official", "A174": "manager/self-employed", "A191": "no telephone", "A192": "telephone",
    }
    return descriptions.get(code, "dataset category")


def latest_profile():
    return st.session_state.get("faircredit_profile")


def render_table(frame: pd.DataFrame, precision: int = 3, percent_columns=()) -> None:
    """Render static tables without Streamlit's Arrow/pyarrow serialization path."""
    frame = frame.copy()
    formatters = {}
    for column in frame.select_dtypes(include="number").columns:
        if column in percent_columns:
            formatters[column] = lambda value: "—" if pd.isna(value) else f"{value:.1%}"
        elif pd.api.types.is_float_dtype(frame[column]):
            formatters[column] = lambda value: "—" if pd.isna(value) else f"{value:.{precision}f}"
    table_html = frame.to_html(
        index=False, escape=True, na_rep="—", border=0,
        classes="fc-table", formatters=formatters,
    )
    st.markdown(f'<div class="fc-table-wrap">{table_html}</div>', unsafe_allow_html=True)


def result_probability(model, profile: pd.DataFrame) -> float:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain bad-credit label 1.")
    return float(model.predict_proba(profile)[0, classes.index(1)])


def model_risk_label(probability: float) -> str:
    if probability < 1 / 3:
        return "Lower estimated probability of the historical bad-credit label"
    if probability < 2 / 3:
        return "Mid-range estimated probability of the historical bad-credit label"
    return "Higher estimated probability of the historical bad-credit label"
