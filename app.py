import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List, Union
from langchain_groq import ChatGroq
import io

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
import os
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Schemas (same as notebook) ──────────────────────────────────────────────

class LabTest(BaseModel):
    name: str
    value: Optional[Union[float, str]] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None

class BloodReport(BaseModel):
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    report_date: Optional[str] = None
    tests: List[LabTest]

class HealthAnalysis(BaseModel):
    overall_summary: str
    concerns: List[str]
    diet_recommendations: List[str]
    exercise_recommendations: List[str]

# ── LLM setup ────────────────────────────────────────────────────────────────

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
extractor = llm.with_structured_output(BloodReport)
advisor   = llm.with_structured_output(HealthAnalysis)

EXTRACTION_PROMPT = """
You are a medical report extraction system.

Extract all patient information and laboratory tests from the report.

Rules:
1. Extract patient name if available.
2. Extract age if available.
3. Extract gender if available.
4. Extract report date if available.

For every laboratory test found:
- test name
- measured value
- unit
- reference range

Do not analyze the report.
Do not provide medical advice.
Do not determine whether values are high or low.
Only extract factual information present in the report.
If a field is missing, return null.

Report:
{report}
"""

# ── File text extraction ──────────────────────────────────────────────────────

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Medical Report Analyzer", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main { padding: 1.5rem 2rem; }
    .hero { text-align: center; padding: 2rem 0 1rem; }
    .hero h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .hero p  { color: #888; font-size: 0.95rem; }
    .divider { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
    .section-title {
        font-size: 1rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; color: #555; margin-bottom: 0.8rem;
    }
    .info-card {
        background: #f8f9fb; border: 1px solid #e4e7ec;
        border-radius: 10px; padding: 0.7rem 1rem;
        display: flex; flex-direction: column; gap: 2px;
    }
    .info-card .label { font-size: 0.72rem; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .info-card .value { font-size: 0.95rem; color: #1a1a2e; font-weight: 600; word-break: break-word; }
    .advice-card {
        border-radius: 10px; padding: 0.75rem 1rem;
        margin-bottom: 0.5rem; font-size: 0.9rem; line-height: 1.5;
    }
    .card-concern  { background: #fff8e1; border-left: 4px solid #f59e0b; color: #78350f; }
    .card-diet     { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }
    .card-exercise { background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e3a5f; }
    .summary-box {
        background: #f0f4ff; border: 1px solid #c7d4f7;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.93rem; color: #1e3a5f; line-height: 1.6;
    }
    .stDataFrame thead tr th { background-color: #f0f4ff !important; font-size: 0.85rem; }
    .stDataFrame tbody tr td { font-size: 0.85rem; }
    .upload-area { max-width: 600px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🩺 Medical Report Analyzer</h1>
    <p>Upload your blood or lab report (PDF, DOCX, or TXT) and get personalized health insights instantly.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload your medical report", type=["pdf", "docx", "txt"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    with st.spinner("Extracting text from report..."):
        report_text = extract_text(uploaded)

    if not report_text.strip():
        st.error("Could not extract text from the file. Please try a different file.")
        st.stop()

    with st.expander("📄 View raw extracted text"):
        st.code(report_text[:3000] + ("..." if len(report_text) > 3000 else ""), language=None)

    with st.spinner("Extracting lab values..."):
        blood_report: BloodReport = extractor.invoke(
            EXTRACTION_PROMPT.format(report=report_text)
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Name",   blood_report.patient_name or "N/A"),
        (c2, "Age",    f"{blood_report.age} yrs" if blood_report.age else "N/A"),
        (c3, "Gender", blood_report.gender or "N/A"),
        (c4, "Date",   blood_report.report_date or "N/A"),
    ]:
        col.markdown(f"""
        <div class="info-card">
            <span class="label">{label}</span>
            <span class="value">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Lab Results</div>', unsafe_allow_html=True)

    if blood_report.tests:
        df = pd.DataFrame([
            {"Test": t.name, "Value": f"{t.value} {t.unit or ''}".strip(), "Reference Range": t.reference_range or "N/A"}
            for t in blood_report.tests
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No lab tests extracted.")

    with st.spinner("Generating health advice..."):
        analysis: HealthAnalysis = advisor.invoke(
            f"""
            Patient Information:
            {blood_report.model_dump_json(indent=2)}

            Generate:
            1. Overall health summary
            2. Main concerns
            3. Diet recommendations
            4. Exercise recommendations

            Do not diagnose diseases.
            """
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Overall Health Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{analysis.overall_summary}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Health Recommendations</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**⚠️ Concerns**")
        for c in analysis.concerns:
            st.markdown(f'<div class="advice-card card-concern">• {c}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**🥗 Diet**")
        for d in analysis.diet_recommendations:
            st.markdown(f'<div class="advice-card card-diet">• {d}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown("**🏃 Exercise**")
        for e in analysis.exercise_recommendations:
            st.markdown(f'<div class="advice-card card-exercise">• {e}</div>', unsafe_allow_html=True)
