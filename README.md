# 🩺 Medical Report Analyzer

A Streamlit web app that accepts a blood/lab report (PDF, DOCX, or TXT) and uses **LLaMA 3.3 70B via Groq** to extract lab values and generate personalized health insights.

---

## ✨ Features

- 📄 Upload PDF, DOCX, or TXT medical reports
- 🔬 Structured extraction of all lab tests (name, value, unit, reference range)
- 👤 Automatic parsing of patient info (name, age, gender, date)
- 💡 AI-generated health advice — summary, concerns, diet & exercise recommendations
- 🎨 Clean, responsive UI built with Streamlit

---

## 🧠 Pipeline

```
Upload File → Extract Text → LLM Extraction (BloodReport schema)
           → LLM Analysis (HealthAnalysis schema) → Display Results
```

Two-step LangChain + Groq pipeline:
1. **Extractor** — pulls structured lab data into a `BloodReport` Pydantic model
2. **Advisor** — generates `HealthAnalysis` with summary, concerns, and recommendations

---

## 🚀 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/Akash-8004/medrx-ai.git
cd medrx-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key-here
```

> Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deployed on Streamlit Cloud : [Live Link](https://medrx-ai-6yaoglfb4fvxz4lzs9wcg4.streamlit.app/)



---

## 📁 Project Structure

```
├── app.py                        # Streamlit app
├── Notebook.ipynb                # Original pipeline notebook
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── secrets.toml.example     # Secrets template (do not commit actual secrets)
├── .env                         # Local env vars (gitignored)
└── .gitignore
```

---

## ⚠️ Disclaimer

This app is for **informational purposes only** and does not provide medical diagnoses. Always consult a qualified healthcare professional for medical advice.
