from uuid import uuid4

import streamlit as st

from faircredit_app.pages.common import input_form, result_probability
from faircredit_app.prediction import validate_profile


def render(context):
    st.markdown('<p class="fc-kicker">FairCredit</p>', unsafe_allow_html=True)
    st.title("Credit assessment")
    st.markdown('<p class="fc-intro">Explore how a historical credit model scores a hypothetical profile. This educational estimate is not a lending decision.</p>', unsafe_allow_html=True)
    with st.form("assessment_form"):
        profile = input_form(context["data"], "assessment")
        submitted = st.form_submit_button("Assess credit risk", type="primary")
    if submitted:
        try:
            st.session_state["faircredit_profile"] = validate_profile(profile, context["data"])
            st.session_state["faircredit_report_id"] = uuid4().hex[:12].upper()
        except ValueError as exc:
            st.error(str(exc))
            return

    profile = st.session_state.get("faircredit_profile")
    if profile is None:
        return
    probability = result_probability(context["training"].model, profile)
    st.session_state["faircredit_probability"] = probability
    predicted_bad = probability >= 0.5
    score = int(round((1 - probability) * 100))
    category = "Low risk" if score >= 67 else "Moderate risk" if score >= 34 else "High risk"
    predicted_label = "Bad-credit history" if predicted_bad else "Good-credit history"
    label_probability = probability if predicted_bad else 1 - probability

    st.subheader("Your estimate")
    score_col, risk_col, label_col = st.columns([1, 1, 1.2])
    score_col.metric("Illustrative risk score", f"{score}/100")
    risk_col.metric("Risk category", category)
    label_col.metric(
        "Probability of predicted dataset label", f"{label_probability:.1%}",
        help=f"Model output label: {predicted_label}",
    )
    st.caption("The score is 100 x (1 - estimated bad-label probability). Higher scores mean a lower model estimate; category bands are illustrative, not lender thresholds.")

    with st.expander("About this estimate"):
        st.write("This model uses historical German Credit data. Its probability is uncalibrated and is not a lender decision, approval likelihood, or professional financial assessment.")
