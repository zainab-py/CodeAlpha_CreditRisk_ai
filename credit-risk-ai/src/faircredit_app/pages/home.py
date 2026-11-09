import streamlit as st


def render(context):
    st.markdown('<p class="fc-kicker">FairCredit</p>', unsafe_allow_html=True)
    st.title("Credit risk assessment")
    st.markdown("Explore an educational model estimate using a hypothetical financial profile.")
    st.caption("Historical-data estimate for information only. Not a bank decision or approval guarantee.")
    left, right = st.columns([1.25, .75])
    with left:
        st.subheader("Explore the app")
        st.markdown("""
1. **Credit assessment** — enter a hypothetical profile and review its estimate.
2. **What-if simulator** — compare the estimate after changing selected inputs.
3. **Research views** — explore model performance and group-level results.
""")
    with right:
        st.subheader("About the data")
        st.metric("Historical records", f"{len(context['data']):,}")
        st.metric("Financial inputs", "17 fields")
        st.caption("Demographic fields are excluded from applicant scoring. No feature-level explanations are provided.")
