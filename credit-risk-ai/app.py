import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CreditRisk AI",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💳 CreditRisk AI")
st.subheader("Explainable Credit Risk Prediction")

st.write(
    "Enter the applicant's financial details below to estimate "
    "their credit risk."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Applicant Details")


# Account status
account_status = st.sidebar.selectbox(
    "Checking Account Status",
    ["A11", "A12", "A13", "A14"]
)

# Loan duration
duration = st.sidebar.slider(
    "Loan Duration (months)",
    min_value=4,
    max_value=72,
    value=24
)

# Credit history
credit_history = st.sidebar.selectbox(
    "Credit History",
    ["A30", "A31", "A32", "A33", "A34"]
)

# Purpose
purpose = st.sidebar.selectbox(
    "Purpose",
    [
        "A40", "A41", "A42", "A43", "A44",
        "A45", "A46", "A47", "A48", "A49", "A410"
    ]
)

# Credit amount
credit_amount = st.sidebar.number_input(
    "Credit Amount",
    min_value=250,
    max_value=20000,
    value=2500,
    step=100
)

# Savings
savings = st.sidebar.selectbox(
    "Savings Account",
    ["A61", "A62", "A63", "A64", "A65"]
)

# Employment
employment = st.sidebar.selectbox(
    "Employment Duration",
    ["A71", "A72", "A73", "A74", "A75"]
)

# Installment rate
installment_rate = st.sidebar.slider(
    "Installment Rate (% of income)",
    1,
    4,
    2
)

# Personal status
personal_status = st.sidebar.selectbox(
    "Personal Status",
    ["A91", "A92", "A93", "A94"]
)

# Other debtors
other_debtors = st.sidebar.selectbox(
    "Other Debtors",
    ["A101", "A102", "A103"]
)

# Property
property_type = st.sidebar.selectbox(
    "Property",
    ["A121", "A122", "A123", "A124"]
)

# Other installment plans
other_installment_plans = st.sidebar.selectbox(
    "Other Installment Plans",
    ["A141", "A142", "A143"]
)

# Housing
housing = st.sidebar.selectbox(
    "Housing",
    ["A151", "A152", "A153"]
)

# Telephone
telephone = st.sidebar.selectbox(
    "Telephone",
    ["A191", "A192"]
)

# Residence
residence = st.sidebar.slider(
    "Years at Current Residence",
    1,
    4,
    2
)

# Age
age = st.sidebar.slider(
    "Age",
    18,
    75,
    30
)

# Existing credits
existing_credits = st.sidebar.slider(
    "Number of Existing Credits",
    1,
    4,
    1
)

# Job
job = st.sidebar.selectbox(
    "Job Type",
    ["A171", "A172", "A173", "A174"]
)

# Dependents
dependents = st.sidebar.slider(
    "Number of Dependents",
    1,
    2,
    1
)


# Foreign worker
foreign_worker = st.sidebar.selectbox(
    "Foreign Worker",
    ["A201", "A202"]
)


# --------------------------------------------------
# CREATE INPUT DATA
# --------------------------------------------------

input_data = pd.DataFrame({
    "checking_status": [account_status],
    "duration": [duration],
    "credit_history": [credit_history],
    "purpose": [purpose],
    "credit_amount": [credit_amount],
    "savings_status": [savings],
    "employment": [employment],
    "installment_rate": [installment_rate],
    "personal_status": [personal_status],
    "other_debtors": [other_debtors],
    "residence_since": [residence],
    "property": [property_type],
    "age": [age],
    "other_installment_plans": [other_installment_plans],
    "housing": [housing],
    "existing_credits": [existing_credits],
    "job": [job],
    "num_dependents": [dependents],
    "telephone": [telephone],
    "foreign_worker": [foreign_worker]
})


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

input_data["credit_to_age_ratio"] = (
    input_data["credit_amount"] / input_data["age"]
)

input_data["duration_to_age_ratio"] = (
    input_data["duration"] / input_data["age"]
)

input_data["monthly_credit_burden"] = (
    input_data["credit_amount"] / input_data["duration"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.sidebar.button("Predict Credit Risk", type="primary"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    bad_probability = probabilities[1]
    good_probability = probabilities[0]

    # Risk score
    risk_score = 100 - (bad_probability * 100)

    # Risk category
    if bad_probability < 0.30:
        risk_category = "Low Risk"
    elif bad_probability < 0.60:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Credit Risk Score",
            f"{risk_score:.1f}/100"
        )

    with col2:
        st.metric(
            "Bad Credit Probability",
            f"{bad_probability * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Risk Category",
            risk_category
        )


    st.divider()


    # --------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Category": ["Good Credit", "Bad Credit"],
        "Probability": [
            good_probability,
            bad_probability
        ]
    })

    fig, ax = plt.subplots()

    ax.bar(
        probability_df["Category"],
        probability_df["Probability"]
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Probability")

    ax.set_title("Credit Classification Probability")

    st.pyplot(fig)


    # --------------------------------------------------
    # APPLICANT PROFILE
    # --------------------------------------------------

    st.subheader("Applicant Profile")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:

        st.write(f"**Age:** {age}")

        st.write(
            f"**Credit Amount:** {credit_amount}"
        )

        st.write(
            f"**Loan Duration:** {duration} months"
        )

        st.write(
            f"**Installment Rate:** {installment_rate}"
        )

    with profile_col2:

        st.write(
            f"**Existing Credits:** {existing_credits}"
        )

        st.write(
            f"**Dependents:** {dependents}"
        )

        st.write(
            f"**Residence Duration:** {residence} years"
        )


    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    st.subheader("Model Interpretation")

    if bad_probability >= 0.60:

        st.error(
            "The model estimates a relatively high probability "
            "of bad credit classification."
        )

    elif bad_probability >= 0.30:

        st.warning(
            "The model estimates a moderate level of credit risk."
        )

    else:

        st.success(
            "The model estimates a relatively low probability "
            "of bad credit classification."
        )


    # --------------------------------------------------
    # WHAT-IF ANALYSIS
    # --------------------------------------------------

    st.divider()

    st.subheader("🔍 What-If Analysis")

    st.write(
        "Change the credit amount below to see how the model's "
        "risk probability changes."
    )

    new_credit_amount = st.slider(
        "What-if Credit Amount",
        min_value=250,
        max_value=20000,
        value=int(credit_amount),
        step=100
    )


    what_if_data = input_data.copy()

    what_if_data["credit_amount"] = new_credit_amount

    what_if_data["credit_to_age_ratio"] = (
        new_credit_amount / age
    )

    what_if_data["monthly_credit_burden"] = (
        new_credit_amount / duration
    )

    what_if_probability = model.predict_proba(
        what_if_data
    )[0][1]

    what_if_score = 100 - (
        what_if_probability * 100
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Original Risk Probability",
            f"{bad_probability * 100:.1f}%"
        )

    with col2:

        st.metric(
            "What-If Risk Probability",
            f"{what_if_probability * 100:.1f}%",
            delta=f"{(what_if_probability - bad_probability) * 100:.1f}%"
        )


    st.write(
        f"**What-if Credit Risk Score:** "
        f"{what_if_score:.1f}/100"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CreditRisk AI is an experimental machine-learning project. "
    "The generated score is model-derived and is not an official "
    "credit score or financial decision."
)