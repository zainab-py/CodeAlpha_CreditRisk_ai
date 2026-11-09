"""FairCredit Streamlit application: educational assessment and research views."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from faircredit import load_german_credit, train_primary_model  # noqa: E402
from faircredit_app.pages import assessment, fairness_insights, home, model_insights, reports, what_if  # noqa: E402

st.set_page_config(page_title="FairCredit | Credit assessment", page_icon="FC", layout="wide")
st.markdown("""
<style>
:root { --ink:#111827; --muted:#344054; --blue:#1d4ed8; --blue-dark:#1e40af; --red:#b42318; --line:#d0d5dd; --paper:#f7f9fc; --white:#fff; }
.stApp { background:var(--paper); color:var(--ink); }
html, body { font-family:Inter,"Segoe UI",sans-serif; color:var(--ink); }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"] { color:var(--ink); }
.block-container { max-width:1080px; padding:4.25rem 2rem 3rem; }
[data-testid="stSidebar"] { min-width:250px; max-width:250px; background:#fff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div { padding:1.35rem 1rem 1rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
[data-testid="stSidebar"] [role="radiogroup"] label { color:var(--ink); font-weight:550; }
[data-testid="stSidebar"] input[type="radio"] { accent-color:var(--blue); }
.fc-brand { display:flex; align-items:center; gap:.65rem; color:var(--ink); font-weight:750; font-size:1.3rem; letter-spacing:-.035em; margin:.15rem 0 .3rem; }
.fc-brand-symbol { display:inline-flex; align-items:center; justify-content:center; width:2rem; height:2rem; color:#fff; background:var(--blue); border-radius:.6rem; font-size:.83rem; letter-spacing:0; position:relative; }
.fc-brand-symbol:after { content:""; position:absolute; width:.42rem; height:.42rem; background:var(--red); border:2px solid #fff; border-radius:50%; right:-.12rem; top:-.12rem; }
h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.025em; }
h1 { font-weight:740; font-size:2.2rem; margin-bottom:.35rem; }
h2 { font-weight:700; font-size:1.45rem; }
h3 { font-weight:680; font-size:1.1rem; }
p, li, label, small, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p { color:var(--muted); line-height:1.6; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p { color:var(--ink); font-weight:550; }
[data-testid="stMetric"] { min-height:108px; border:1px solid var(--line); border-top:3px solid var(--blue); background:var(--white); padding:1rem 1.1rem; border-radius:12px; box-shadow:0 3px 12px rgba(16,24,40,.045); }
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.85rem; }
[data-testid="stMetricValue"] { color:var(--ink) !important; font-weight:740; }
[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.35rem 1.45rem; box-shadow:0 5px 22px rgba(16,24,40,.04); }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  color:var(--ink) !important;
  -webkit-text-fill-color:var(--ink);
  background:#fff !important;
  opacity:1;
}
[data-testid="stSelectbox"] [data-baseweb="select"] {
  color:var(--ink) !important;
  background:#fff !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  color:var(--ink) !important;
  background:#fff !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
  color:var(--ink) !important;
  -webkit-text-fill-color:var(--ink);
  background:transparent !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] div[value],
[data-testid="stSelectbox"] [role="combobox"] {
  color:var(--ink) !important;
  -webkit-text-fill-color:var(--ink);
  background:transparent !important;
}
[data-testid="stSelectbox"] [role="combobox"] { min-height:2.8rem; }
[data-testid="stNumberInput"] input::placeholder, [data-testid="stTextInput"] input::placeholder { color:#475467 !important; opacity:1; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  color:var(--ink) !important;
  background:#fff !important;
}
[data-baseweb="popover"] [role="option"], [role="listbox"] [role="option"], [role="option"] {
  color:var(--ink) !important;
  -webkit-text-fill-color:var(--ink);
  background:#fff !important;
}
[data-baseweb="popover"] [role="option"] *, [role="listbox"] [role="option"] *, [role="option"] * {
  color:var(--ink) !important;
  -webkit-text-fill-color:var(--ink);
  background:transparent !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[role="listbox"] [role="option"]:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"] {
  color:var(--ink) !important;
  background:#eaf0ff !important;
}
button[kind="primary"] { background:var(--blue); border:1px solid var(--blue); border-radius:9px; font-weight:650; min-height:2.8rem; }
button[kind="primary"]:hover { background:var(--blue-dark); border-color:var(--blue-dark); }
button[kind="primary"] p { color:#fff !important; }
button:focus-visible, a:focus-visible, input:focus-visible, [role="combobox"]:focus-visible { outline:3px solid #84adff !important; outline-offset:2px; }
a { color:var(--blue); }
a:hover { color:var(--red); }
.fc-kicker { color:var(--red) !important; font-weight:750; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; margin:0 0 .4rem; }
.fc-intro { max-width:760px; margin-bottom:1.2rem; }
.fc-section { border-bottom:1px solid var(--line); border-left:3px solid var(--blue); padding:.45rem 0 .6rem .7rem; margin:1.1rem 0 .65rem; color:var(--ink); font-weight:720; font-size:1rem; }
.fc-note { color:var(--muted); font-size:.88rem; margin-top:-.2rem; margin-bottom:.8rem; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:9px; overflow:hidden; }
.fc-table-wrap { max-width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:#fff; margin:.25rem 0 .8rem; }
.fc-table { width:100%; border-collapse:collapse; color:var(--ink); font-size:.9rem; }
.fc-table th { background:#eaf0ff; color:#101828; font-weight:700; text-align:left; white-space:nowrap; }
.fc-table th,.fc-table td { padding:.62rem .75rem; border-bottom:1px solid #eaecf0; }
.fc-table tr:last-child td { border-bottom:0; }
.fc-table td { color:var(--ink); }
@media (max-width: 760px) {
  .block-container { padding:3.75rem 1rem 2rem; }
  h1 { font-size:1.8rem; }
  [data-testid="stForm"] { padding:1rem; border-radius:14px; }
  [data-testid="stSidebar"] { min-width:unset; max-width:unset; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_german_credit()


@st.cache_resource
def get_training():
    return train_primary_model(get_data())


pages = {
    "Assessment": assessment.render,
    "Overview": home.render,
    "What-if simulator": what_if.render,
    "Fairness insights": fairness_insights.render,
    "Model insights": model_insights.render,
    "Report": reports.render,
}
st.sidebar.markdown('<div class="fc-brand"><span class="fc-brand-symbol">FC</span>FairCredit</div>', unsafe_allow_html=True)
st.sidebar.caption("Credit risk assessment")
selected_page = st.sidebar.radio("Navigate", list(pages), index=0, label_visibility="collapsed")

try:
    data = get_data()
    training = get_training()
except Exception as exc:
    st.error(f"FairCredit could not load its bundled dataset or fit the demonstration model: {exc}")
    st.info("Check that credit-risk-ai/data/german.data is present and install requirements.txt.")
    st.stop()

context = {"data": data, "training": training, "experiment_root": ROOT / "reports" / "experiments"}
try:
    pages[selected_page](context)
except Exception as exc:
    st.error(f"This view could not be loaded: {exc}")
    st.caption("Review the local data and dependency setup, then restart the application.")

st.divider()
st.caption("Historical-data estimate for information only. Not a bank decision or approval guarantee.")
