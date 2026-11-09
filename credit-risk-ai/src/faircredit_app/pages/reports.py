from uuid import uuid4

import streamlit as st

from faircredit_app.pages.common import latest_profile, result_probability
from faircredit_app.reporting import DISCLAIMER, generate_assessment_pdf


def render(context):
    st.markdown('<p class="fc-kicker">Privacy-first, downloadable report</p>', unsafe_allow_html=True)
    st.title("Assessment report")
    st.markdown(
        '<p class="fc-intro">A clear summary of the model estimate, how to interpret it, '
        'and how the model performed. Your entered financial details are not included.</p>',
        unsafe_allow_html=True,
    )
    profile = latest_profile()
    if profile is None:
        st.info("Complete a Credit assessment first to prepare a report.")
        return

    try:
        training = context["training"]
        bad_probability = result_probability(training.model, profile)
        score = int(round((1 - bad_probability) * 100))
        category = "Low risk" if score >= 67 else "Moderate risk" if score >= 34 else "High risk"
        predicted_label = (
            "Bad-credit history (dataset label 1)"
            if bad_probability >= 0.5
            else "Good-credit history (dataset label 0)"
        )
        report_id = st.session_state.setdefault(
            "faircredit_report_id", uuid4().hex[:12].upper(),
        )
        pdf = generate_assessment_pdf(
            report_id=report_id,
            bad_probability=bad_probability,
            risk_score=score,
            risk_category=category,
            predicted_label=predicted_label,
            dataset_rows=len(context["data"]),
            training_rows=training.train_rows,
            test_rows=training.test_rows,
            holdout_metrics=training.metrics,
            model_parameters=training.selected_parameters,
        )
    except Exception as exc:
        st.error(f"Could not prepare the assessment report: {exc}")
        return

    st.caption(f"Report ID: {report_id}  ·  Based on the latest submitted hypothetical assessment")
    score_col, risk_col, probability_col = st.columns(3)
    score_col.metric("Illustrative score", f"{score}/100")
    risk_col.metric("Risk band", category)
    probability_col.metric("Estimated bad-label probability", f"{bad_probability:.1%}")

    st.subheader("What the PDF includes")
    with st.container(border=True):
        st.markdown(
            "- Score, risk band, predicted historical dataset label, and bad-label probability\n"
            "- A plain-language explanation of the score and category thresholds\n"
            "- Holdout accuracy, precision, recall, F1, and ROC-AUC with brief definitions\n"
            "- Model setup, dataset size, validation split, and important limitations"
        )
        st.info(
            "Privacy: the PDF does not contain your entered financial fields, identity details, "
            "or demographic information. It is generated in memory; downloaded files are saved by your browser."
        )

    st.download_button(
        "Download detailed assessment report (PDF)",
        data=pdf,
        file_name=f"faircredit_assessment_{report_id}.pdf",
        mime="application/pdf",
        type="primary",
    )
    st.caption(DISCLAIMER)
