from uuid import uuid4

import streamlit as st

from faircredit_app.pages.common import code_description, latest_profile, result_probability


def render(context):
    st.markdown('<p class="fc-kicker">Hypothetical scenarios</p>', unsafe_allow_html=True)
    st.title("What-if simulator")
    st.write("Change selected financial inputs to compare model estimates. Demographic attributes are not used in applicant scoring.")
    original = latest_profile()
    if original is None:
        st.info("Complete a Credit assessment first to create a hypothetical profile.")
        return
    model = context["training"].model
    before = result_probability(model, original)
    col_a, col_b, col_c = st.columns(3)
    amount_values = context["data"]["credit_amount"]
    duration_values = context["data"]["duration"]
    with col_a:
        amount = st.number_input(
            "Alternative requested amount (DM)",
            int(amount_values.min()), int(amount_values.max()), int(original.iloc[0]["credit_amount"]), 100,
        )
    with col_b:
        duration = st.slider(
            "Alternative loan duration (months)",
            int(duration_values.min()), int(duration_values.max()), int(original.iloc[0]["duration"]),
        )
    with col_c:
        employment_options = sorted(context["data"]["employment"].dropna().astype(str).unique())
        employment = st.selectbox(
            "Employment duration",
            employment_options,
            index=employment_options.index(str(original.iloc[0]["employment"])),
            format_func=lambda code: f"{code} · {code_description(code)}",
        )
    scenario = original.copy()
    scenario.loc[0, "credit_amount"] = amount
    scenario.loc[0, "duration"] = duration
    scenario.loc[0, "employment"] = employment
    scenario.loc[0, "principal_per_month_proxy"] = amount / duration
    after = result_probability(model, scenario)
    left, right = st.columns(2)
    left.metric("Original model estimate", f"{before:.1%}")
    right.metric("Changed scenario estimate", f"{after:.1%}", f"{(after-before)*100:+.1f} percentage points")
    st.caption("Principal per month is amount divided by duration; it is not a repayment estimate because interest, fees, and income are absent. A changed result is not a causal recommendation or guaranteed route to credit approval.")
    if st.button("Use this scenario for assessment", type="primary"):
        st.session_state["faircredit_profile"] = scenario
        st.session_state["faircredit_probability"] = after
        st.session_state["faircredit_report_id"] = uuid4().hex[:12].upper()
        st.success("Scenario saved as the current hypothetical profile. Return to Credit assessment to view its score.")
