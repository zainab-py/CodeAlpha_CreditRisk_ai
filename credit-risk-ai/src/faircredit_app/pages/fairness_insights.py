import json
import pandas as pd
import streamlit as st

from faircredit_app.pages.common import render_table

GROUPS = {"Age bands": "age_group", "Personal status/sex code": "personal_status", "Foreign-worker status": "foreign_worker"}


def render(context):
    st.markdown('<p class="fc-kicker">Population-level research</p>', unsafe_allow_html=True)
    st.title("Fairness insights")
    st.write("These aggregate comparisons examine whether prediction rates and error rates differ across demographic groups in the public historical dataset.")
    st.info("This page shows population-level research results only. It does not display another applicant's personal information.")
    label = st.selectbox("Audit attribute", list(GROUPS))
    folder = context["experiment_root"] if label == "Age bands" else context["experiment_root"] / GROUPS[label]
    result_path = folder / "experiment_results.json"
    if not result_path.exists():
        st.warning("No saved fairness results were found. Run experiments.py to generate the research artifacts.")
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    models = result["models_and_experiments"]
    selected = st.selectbox("Model", list(models), key="fairness_model")
    fairness = models[selected]["fairness"]
    by_group = pd.DataFrame(fairness["by_group"])
    favorable = pd.DataFrame(fairness["favorable_by_group"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Bad-credit error and performance rates by group**")
        render_table(by_group, percent_columns=("Predicted high risk", "True positive rate", "False positive rate"))
    with right:
        st.markdown("**Favorable selection rate by group**")
        render_table(favorable, percent_columns=("favorable_selection_rate_good", "true_positive_rate_good", "false_positive_rate_good"))
    summary = fairness.get("summary", {})
    metrics = pd.DataFrame([
        {"Metric": "Demographic parity difference", "Value": summary.get("demographic_parity_difference"), "Meaning": "Gap in favorable prediction rates"},
        {"Metric": "Demographic parity ratio", "Value": summary.get("demographic_parity_ratio"), "Meaning": "Minimum favorable rate divided by maximum"},
        {"Metric": "Equal opportunity difference", "Value": summary.get("equal_opportunity_difference"), "Meaning": "Gap in favorable true-positive rates"},
        {"Metric": "Equalized odds difference", "Value": summary.get("equalized_odds_difference"), "Meaning": "Largest favorable TPR/FPR gap"},
        {"Metric": "Equalized odds favorable TPR gap", "Value": summary.get("equalized_odds_tpr_good_difference"), "Meaning": "Gap in true-positive rates for good credit"},
        {"Metric": "Equalized odds favorable FPR gap", "Value": summary.get("equalized_odds_fpr_good_difference"), "Meaning": "Gap in false-positive rates for good credit"},
    ])
    st.markdown("**Summary metrics**")
    render_table(metrics)
    st.caption("Favorable means predicted good credit (class 0). Recall/error metrics treat bad credit (class 1) as positive. Group sizes and small-group warnings are shown in the underlying tables.")
    st.write("Fairness measures can conflict. Results depend on historical labels, group sizes, the decision threshold, and metric choice; a small gap does not establish that a model is unbiased.")
    comparison_path = folder / "model_comparison.csv"
    if comparison_path.exists():
        table = pd.read_csv(comparison_path)
        relevant = [column for column in ["Model", "Accuracy", "F1", "ROC-AUC", "DP Difference", "EO Difference", "EOdds Difference", "Mitigation"] if column in table]
        st.markdown("**Baseline and mitigation comparisons**")
        render_table(table[relevant])
        st.caption("Reweighing and constrained optimization are experiments. Improvements in one metric can coincide with worse results in another.")
