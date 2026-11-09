import json
import pandas as pd
import streamlit as st

from faircredit_app.pages.common import render_table


def render(context):
    st.markdown('<p class="fc-kicker">Research workspace</p>', unsafe_allow_html=True)
    st.title("Model insights")
    st.write("Predictive performance details from the fixed holdout experiment. This section is separate from the applicant assessment.")
    folder = context["experiment_root"]
    result_path, csv_path = folder / "experiment_results.json", folder / "model_comparison.csv"
    if csv_path.exists():
        comparison = pd.read_csv(csv_path)
        st.subheader("Model comparison")
        render_table(comparison)
    if result_path.exists():
        results = json.loads(result_path.read_text(encoding="utf-8"))
        models = results["models_and_experiments"]
        selected = st.selectbox("Holdout diagnostics", list(models))
        details = models[selected]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix Â· rows actual, columns predicted**")
            matrix = pd.DataFrame(details["confusion_matrix"], index=["Actual good", "Actual bad"], columns=["Predicted good", "Predicted bad"])
            matrix.insert(0, "Actual", matrix.index)
            render_table(matrix.reset_index(drop=True))
        with c2:
            st.markdown("**Classification report**")
            st.json(details["classification_report"], expanded=False)
        for filename in ["model_curves_and_fairness.png", "confusion_matrices.png", "confusion_and_group_rates.png", "threshold_sensitivity.png"]:
            image = folder / filename
            if image.exists():
                st.image(str(image), caption=filename.replace("_", " ").removesuffix(".png"))
        st.caption("Curves and calibration values are holdout estimates. They describe this split and do not certify future performance.")
    else:
        st.info("Run experiments.py to generate ROC, precision-recall, calibration, and fairness result artifacts.")
