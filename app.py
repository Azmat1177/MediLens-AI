"""MediLens AI - Streamlit UI.  Run:  streamlit run app.py
Needs medilens_agents.py in the same folder."""
import os
import tempfile

import streamlit as st

# Streamlit Cloud: copy the secret into an env var BEFORE importing the agents
try:
    for _k in ("GEMINI_API_KEY", "MEDILENS_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally is fine

from medilens_agents import DRUG_KB, build_graph

st.set_page_config(page_title="MediLens AI", page_icon="🩺", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


SAMPLE_CBC = """Hemoglobin 9.8 g/dL (12.0-15.5)
MCV 72 fL (80-100)
WBC 7.2 x10^3/uL (4.0-11.0)
Platelets 250 x10^3/uL (150-450)"""


def save_upload(f) -> str:
    suffix = os.path.splitext(f.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.getbuffer())
        return tmp.name


def show_result(out: dict):
    if out.get("needs_user") or out.get("blocked"):
        st.warning(out["final_answer"])
    elif out.get("escalate") or out.get("emergency"):
        st.error(out["final_answer"])
    else:
        st.success("Analysis complete")
        st.markdown(out["final_answer"])
    with st.expander("Agent trace (for developers)"):
        for step in out.get("trace", []):
            st.code(step, language=None)
        if out.get("verification"):
            st.json(out["verification"])


st.title("🩺 MediLens AI")
st.caption("Understand your medical reports and medicines in simple language. Not medical advice.")

with st.sidebar:
    st.header("Settings")
    lang_label = st.radio("Explanation language", ["English", "Roman Urdu"])
    language = "en" if lang_label == "English" else "roman_urdu"
    sex = st.selectbox("Patient sex (for reference ranges)", ["F", "M"])
    age = st.number_input("Age", 1, 120, 30)
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        st.success("Gemini key detected. Photo/PDF reading is on.")
    else:
        st.info("No Gemini key found. Paste report text or type the medicine name, "
                "or add GEMINI_API_KEY in Secrets.")

app = get_graph()
tab_report, tab_med, tab_ask = st.tabs(["📄 Medical report", "💊 Medicine", "💬 Question"])

with tab_report:
    up = st.file_uploader("Upload report (image or PDF)", type=["png", "jpg", "jpeg", "pdf"], key="rep")
    text = st.text_area("Or paste report text", value=SAMPLE_CBC, height=140)
    if st.button("Analyze report", type="primary"):
        state = {"language": language, "patient": {"sex": sex, "age": age}}
        if up:  # uploaded file wins; Gemini reads it (or pytesseract fallback)
            state["file_path"] = save_upload(up)
        elif text.strip():
            state["raw_text"] = text
        with st.spinner("Agents are working..."):
            show_result(app.invoke(state))

with tab_med:
    img = st.file_uploader("Upload medicine photo", type=["png", "jpg", "jpeg"], key="med")
    if img:
        st.image(img, width=250)
    c1, c2 = st.columns(2)
    name = c1.text_input("Medicine name (confirm what you see)", placeholder="e.g. Panadol")
    strength = c2.text_input("Strength", placeholder="e.g. 500mg")
    st.caption("Known in demo: " + ", ".join(k.title() for k in DRUG_KB))
    if st.button("Identify medicine", type="primary"):
        state = {"language": language}
        if name:  # typed name = user-confirmed, skips vision
            state["medicine_hint"] = {"name": name, "strength": strength, "confidence": 0.9}
        elif img:
            state["file_path"] = save_upload(img)  # Gemini vision reads the photo
        if "medicine_hint" not in state and "file_path" not in state:
            st.warning("Upload a photo or type the medicine name.")
        else:
            with st.spinner("Medicine agent is working..."):
                show_result(app.invoke(state))

with tab_ask:
    q = st.text_input("Ask a question (safety check demo)", placeholder="Can you change my dose?")
    if st.button("Send") and q:
        out = app.invoke({"language": language, "user_text": q, "raw_text": ""})
        show_result(out)
