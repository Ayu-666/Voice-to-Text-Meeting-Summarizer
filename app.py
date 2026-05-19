import streamlit as st
from groq import Groq
import os

st.set_page_config(page_title="AI Meeting Summarizer", page_icon="🎙️")
st.title("🎙️ Voice-to-Text Meeting Summarizer")
st.write("Convert meeting recordings into structured minutes, action items, and summaries instantly.")

# 1. Secure API Key Loading
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# Initialize the official native Groq client
client = Groq(api_key=api_key)

# 2. Audio File Uploader UI
uploaded_file = st.file_uploader(
    "Upload meeting audio recording", 
    type=["mp3", "wav", "m4a", "webm", "ogg"]
)

if uploaded_file:
    # Display an audio player widget so the user can review the upload
    st.audio(uploaded_file, format="audio/mp3")
    
    # Save uploaded file temporarily to disk for the API reader
    temp_audio_path = f"temp_{uploaded_file.name}"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create layout tabs for clean portfolio presentation
    tab1, tab2 = st.tabs(["📝 Transcript", "📊 Structured Summary"])

    # 3. Audio Transcription Stage (Whisper Large V3)
    if "transcript" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Transcribing audio using Whisper-Large-V3..."):
            try:
                with open(temp_audio_path, "rb") as audio_file:
                    # Request cloud speech-to-text transcription
                    transcription = client.audio.transcriptions.create(
                        file=(temp_audio_path, audio_file.read()),
                        model="whisper-large-v3",
                        response_format="text"
                    )
                # Cache transcription and track current filename in session state
                st.session_state.transcript = transcription
                st.session_state.file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Transcription API Error: {e}")
                st.stop()

    with tab1:
        st.subheader("Raw Meeting Transcript")
        st.write(st.session_state.transcript)

    # 4. LLM Summary & Insights Generation Stage (Llama 3.1)
    with tab2:
        st.subheader("AI Generated Meeting Minutes")
        
        if st.button("Generate Summary & Action Items 🚀"):
            with st.spinner("Analyzing text and formatting business insights..."):
                # System instructions prompting specific executive formatting styles
                system_prompt = (
                    "You are an expert corporate secretary. Your task is to analyze the following meeting transcript "
                    "and provide a highly structured, professional summary. You MUST format the output exactly into "
                    "the following markdown sections:\n\n"
                    "### 🎯 Executive Summary\n(A concise paragraph outlining the main purpose and outcome of the meeting)\n\n"
                    "### 🔑 Key Discussion Points\n(Bullet points highlighting important topics addressed)\n\n"
                    "### ⚠️ Action Items & Ownership\n(Numbered list specifying tasks, deadlines, and assigned personnel if mentioned)\n\n"
                    "Ensure your tone is sharp, objective, and executive-ready."
                )
                
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Transcript:\n{st.session_state.transcript}"}
                        ],
                        model="llama-3.1-8b-instant",
                        temperature=0.3 # Low temperature ensures focused, factual summaries
                    )
                    
                    # Output the beautifully structured markdown directly to the tab
                    st.markdown(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Inference Engine Error: {e}")

    # Clean up the local disk space
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)