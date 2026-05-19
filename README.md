# 🎙️ Voice-to-Text Meeting Summarizer

A production-grade asynchronous corporate intelligence application designed to automate the generation of structured meeting documentation from raw audio. This application ingests audio recordings, performs rapid cloud-hosted speech-to-text transcription, and processes the text through an intelligent LLM synthesis engine to produce clear, actionable corporate notes.

By offloading both audio transcription and linguistic modeling to Groq's high-speed hardware APIs, the tool maintains zero local performance overhead and operates seamlessly without complex system-level audio dependencies (like `ffmpeg`).

## 🚀 Features
* **Asynchronous Voice Parsing:** Ingests variable-length compressed audio file formats (`.mp3`, `.wav`, `.m4a`, `.webm`, `.ogg`).
* **High-Fidelity Cloud ASR:** Utilizes **Whisper-Large-V3** via the native Groq SDK for accurate speech recognition and punctuation.
* **Corporate Synthesis Framework:** Leverages low-temperature **Llama 3.1 Inference** to systematically extract business value while preventing factual hallucinations.
* **Executive Document Formatter:** Automatically maps unstructured conversations into definitive enterprise modules:
  * 🎯 **Executive Summary:** High-level strategic overview.
  * 🔑 **Key Discussion Points:** Structured topical vectors.
  * ⚠️ **Action Items & Ownership:** Explicit tasks matched with personnel and target goals.
* **Dual-Tab Developer Presentation:** Features a modular UI to view raw transcriptional data alongside completed analytical summaries.

## 🛠️ Tech Stack
* **UI Interface:** [Streamlit](https://streamlit.io/)
* **Orchestration Client:** [Groq Native SDK](https://github.com/groq/groq-python)
* **Speech-to-Text Engine:** [Whisper-Large-V3](https://openai.com/research/whisper) (Cloud-hosted)
* **Text Processing Model:** [Llama-3.1-8b-instant](https://meta.ai/)

## 💻 Local Setup & Deployment

**1. Clone the repository**
```bash
git clone [https://github.com/MohibAhmadButt/Voice-to-Text-Meeting-Summarizer.git](https://github.com/MohibAhmadButt/Voice-to-Text-Meeting-Summarizer.git)
cd Voice-to-Text-Meeting-Summarizer
```
**2. Isolate your virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**3. Install dependencies**
```Bash
pip install -r requirements.txt
```
**4. Setup Local Secret Key**

Create a .streamlit/secrets.toml file matching your configuration rules:

GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"

**5. Launch the application**
streamlit run app.py
```bash
streamlit run app.py
```
