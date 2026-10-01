import json
from datetime import date

import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="BBIT Summarizer", layout="wide")
st.title("Smart Meeting & Lecture Summarizer (AI-03)")
st.info("🔒 This app doesn't save audio. It is sent to Groq for transcription and analysis.")

WHISPER = "whisper-large-v3-turbo"
LLMS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MAX_MB = 25
COLS = ["Task Description", "Assigned Person", "Priority", "Implied Deadline"]

if "result" not in st.session_state:
    st.session_state.result = None

api_key = st.text_input("Enter your Groq API Key to start:", type="password")

st.subheader("1. Input Sources")
mic_audio = st.audio_input("🎙️ Record Live Mic")
file_audio = st.file_uploader("📁 Or Upload MP3/WAV", type=["mp3", "wav"])
raw_text = st.text_area("📝 Or Paste Transcript Here", height=150)


def process(client, transcript, audio_src):
    if audio_src is not None:
        if audio_src.size > MAX_MB * 1024 * 1024:
            raise ValueError(f"Audio is over the {MAX_MB} MB limit.")
        name = getattr(audio_src, "name", None) or "recording.wav"
        transcript = client.audio.transcriptions.create(
            file=(name, audio_src.getvalue()), model=WHISPER
        ).text
    if not transcript or not transcript.strip():
        raise ValueError("No speech/text found. Check your mic or input and try again.")

    prompt = f"""Analyze this transcript and return ONLY valid JSON in this exact schema:
{{
  "executive_summary": ["bullet 1", "bullet 2", "bullet 3"],
  "decisions": ["decision 1"],
  "tasks": [{{"Task Description": "desc", "Assigned Person": "name or Unassigned",
             "Priority": "High|Medium|Low",
             "Implied Deadline": "YYYY-MM-DD or null"}}]
}}
Rules: use only information in the transcript, never invent tasks or names.
Today is {date.today().isoformat()}. Give a deadline date ONLY if the transcript clearly states
one (e.g. "by Friday"); otherwise use null.
Transcript: {transcript[:30000]}"""

    last_err = None
    for model in LLMS:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return transcript, json.loads(res.choices[0].message.content)
        except Exception as e:  # try the next model
            last_err = e
    raise last_err


if st.button("✨ Generate Minutes & Tasks", type="primary"):
    audio_src = mic_audio or file_audio
    if not api_key:
        st.error("Please enter your Groq API Key above.")
    elif audio_src is None and not raw_text.strip():
        st.error("Please provide an audio recording or a text transcript.")
    else:
        try:
            with st.spinner("Transcribing and analysing..."):
                transcript, data = process(Groq(api_key=api_key), raw_text, audio_src)
            df = pd.DataFrame(data.get("tasks") or [], columns=COLS).fillna("")
            df.insert(0, "Done", False)
            st.session_state.result = {"transcript": transcript, "data": data, "df": df}
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

res = st.session_state.result
if res:
    data = res["data"]
    st.success("✅ Analysis complete!")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Executive 3-Bullet Summary")
        for b in data.get("executive_summary", []):
            st.write(f"• {b}")
    with c2:
        st.subheader("Key Decisions")
        for d in data.get("decisions", []) or ["None detected"]:
            st.write(f"• {d}")

    st.subheader("Action Items (editable)")
    if res["df"].empty:
        st.write("No action items detected.")
        edited = res["df"]
    else:
        edited = st.data_editor(res["df"], hide_index=True, key="tasks_editor")

    md = ["# Meeting Minutes", "", "## Summary"]
    md += [f"- {b}" for b in data.get("executive_summary", [])]
    md += ["", "## Decisions"] + [f"- {d}" for d in data.get("decisions", [])]
    md += ["", "## Action Items"]
    for _, r in edited.iterrows():
        md.append(f"- [{'x' if r['Done'] else ' '}] {r['Task Description']} "
                  f"({r['Assigned Person']}, {r['Priority']}, due: {r['Implied Deadline'] or '-'})")
    md += ["", "## Transcript", res["transcript"]]
    st.download_button("📥 Download Markdown", "\n".join(md), "minutes.md", "text/markdown")

    with st.expander("Transcript"):
        st.write(res["transcript"])
