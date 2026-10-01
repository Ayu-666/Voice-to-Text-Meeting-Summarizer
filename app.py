 import json
import pandas as pd
import streamlit as st
from groq import Groq

# ==========================================
# 1. PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="BBIT Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Smart Meeting & Lecture Summarizer")
st.caption("AI-powered transcription, summaries, decisions and action items.")

st.info("🔒 Audio is sent to Groq for transcription. No audio is permanently stored by this app.")

# ==========================================
# 2. API KEY
# ==========================================

api_key = st.text_input(
    "Enter your Groq API Key:",
    type="password",
    placeholder="gsk_..."
)

# ==========================================
# 3. INPUT SOURCES
# ==========================================

st.subheader("1. Input Source")

mic_audio = st.audio_input("🎙️ Record from microphone")

file_audio = st.file_uploader(
    "📁 Or upload an audio file",
    type=["mp3", "wav", "m4a", "webm"]
)

raw_text = st.text_area(
    "📝 Or paste a transcript",
    height=180,
    placeholder="Paste your meeting or lecture transcript here..."
)

# ==========================================
# 4. MODELS
# ==========================================

# Current Groq Whisper model
WHISPER_MODEL = "whisper-large-v3"

# Use a currently available Groq Llama model.
# If Groq changes availability, this is the only
# line you need to change.
LLM_MODEL = "openai/gpt-oss-20b"

# ==========================================
# 5. MAIN BUTTON
# ==========================================

if st.button(
    "✨ Generate Minutes & Tasks",
    type="primary",
    use_container_width=True
):

    # --------------------------------------
    # Validate API key
    # --------------------------------------

    if not api_key:
        st.error("❌ Please enter your Groq API key.")
        st.stop()

    # --------------------------------------
    # Validate input
    # --------------------------------------

    audio_src = mic_audio if mic_audio is not None else file_audio

    if audio_src is None and not raw_text.strip():
        st.error("❌ Please provide an audio file, microphone recording, or transcript.")
        st.stop()

    try:

        client = Groq(api_key=api_key)

        transcript = raw_text.strip()

        # ==================================
        # 6. TRANSCRIPTION
        # ==================================

        if audio_src is not None:

            with st.spinner("🎙️ Transcribing audio..."):

                audio_bytes = audio_src.getvalue()

                if not audio_bytes:
                    st.error("❌ The audio file appears to be empty.")
                    st.stop()

                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", audio_bytes),
                    model=WHISPER_MODEL,
                    response_format="text"
                )

                # Groq may return either a string or an object
                if isinstance(transcription, str):
                    transcript = transcription.strip()
                else:
                    transcript = getattr(
                        transcription,
                        "text",
                        str(transcription)
                    ).strip()

        # ==================================
        # 7. CHECK TRANSCRIPT
        # ==================================

        if not transcript:

            st.error(
                "❌ No speech/transcript was detected. "
                "Try a clearer recording or paste the transcript."
            )

            st.stop()

        # Show transcript for debugging / transparency

        with st.expander("📄 View Transcript"):
            st.write(transcript)

        # ==================================
        # 8. LLM PROMPT
        # ==================================

        prompt = f"""
You are a professional meeting and lecture assistant.

Analyze the transcript below.

Return ONLY valid JSON.
Do NOT use Markdown.
Do NOT put the JSON inside ``` code fences.

Use EXACTLY this structure:

{{
  "executive_summary": [
    "bullet 1",
    "bullet 2",
    "bullet 3"
  ],
  "decisions": [
    "decision 1"
  ],
  "tasks": [
    {{
      "Task Description": "description",
      "Assigned Person": "person or Unassigned",
      "Priority": "High",
      "Deadline": "YYYY-MM-DD or Not specified"
    }}
  ]
}}

IMPORTANT RULES:

- Give exactly 3 concise executive summary bullets when enough information exists.
- Only include decisions that are actually stated or clearly agreed upon.
- Only create tasks that are actually present in the transcript.
- Never invent people.
- If no person is assigned, use "Unassigned".
- If no deadline is mentioned, use "Not specified".
- Priority must be exactly one of:
  High
  Medium
  Low
- Do not invent dates.
- If there are no decisions, return an empty array.
- If there are no tasks, return an empty array.

TRANSCRIPT:

{transcript}
"""

        # ==================================
        # 9. LLM ANALYSIS
        # ==================================

        with st.spinner("🧠 Analyzing transcript..."):

            chat_res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2000
            )

        # ==================================
        # 10. GET MODEL RESPONSE
        # ==================================

        content = chat_res.choices[0].message.content.strip()

        # Remove accidental Markdown fences

        if content.startswith("```"):
            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        # ==================================
        # 11. PARSE JSON
        # ==================================

        try:

            data = json.loads(content)

        except json.JSONDecodeError:

            # Try extracting JSON object
            start = content.find("{")
            end = content.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    "The AI did not return valid JSON.\n\n"
                    f"AI response:\n{content}"
                )

            json_text = content[start:end + 1]

            try:
                data = json.loads(json_text)

            except json.JSONDecodeError as json_error:

                raise ValueError(
                    "AI returned malformed JSON.\n\n"
                    f"AI response:\n{content}\n\n"
                    f"JSON error:\n{json_error}"
                )

        # ==================================
        # 12. NORMALIZE DATA
        # ==================================

        if not isinstance(data, dict):
            raise ValueError("AI response was not a JSON object.")

        summary = data.get("executive_summary", [])
        decisions = data.get("decisions", [])
        tasks = data.get("tasks", [])

        if not isinstance(summary, list):
            summary = []

        if not isinstance(decisions, list):
            decisions = []

        if not isinstance(tasks, list):
            tasks = []

        # ==================================
        # 13. SUCCESS
        # ==================================

        st.success("✅ Analysis complete!")

        # ==================================
        # 14. SUMMARY + DECISIONS
        # ==================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📌 Executive Summary")

            if summary:

                for item in summary:
                    st.markdown(f"• {item}")

            else:
                st.write("No summary available.")

        with col2:

            st.subheader("✅ Key Decisions")

            if decisions:

                for item in decisions:
                    st.markdown(f"• {item}")

            else:
                st.write("No decisions detected.")

        # ==================================
        # 15. ACTION ITEMS
        # ==================================

        st.subheader("📋 Action-Item Matrix")

        if tasks:

            df = pd.DataFrame(tasks)

            # Make sure expected columns exist

            expected_columns = [
                "Task Description",
                "Assigned Person",
                "Priority",
                "Deadline"
            ]

            for column in expected_columns:

                if column not in df.columns:
                    df[column] = "Not specified"

            df = df[expected_columns]

            df.insert(0, "Done", False)

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Done": st.column_config.CheckboxColumn(
                        "Done"
                    )
                }
            )

        else:

            st.write("No action items detected.")

        # ==================================
        # 16. EXPORT
        # ==================================

        st.subheader("📥 Export")

        md_export = "# Meeting Minutes\n\n"

        md_export += "## Executive Summary\n\n"

        for item in summary:
            md_export += f"- {item}\n"

        md_export += "\n## Key Decisions\n\n"

        for item in decisions:
            md_export += f"- {item}\n"

        md_export += "\n## Action Items\n\n"

        if tasks:

            for task in tasks:

                md_export += (
                    f"- **Task:** {task.get('Task Description', 'Not specified')}\n"
                    f"  - **Assigned:** {task.get('Assigned Person', 'Unassigned')}\n"
                    f"  - **Priority:** {task.get('Priority', 'Medium')}\n"
                    f"  - **Deadline:** {task.get('Deadline', 'Not specified')}\n\n"
                )

        else:

            md_export += "No action items detected.\n"

        st.download_button(
            label="📥 Download Markdown",
            data=md_export,
            file_name="meeting_minutes.md",
            mime="text/markdown",
            use_container_width=True
        )

    # ======================================
    # 17. ERROR HANDLING
    # ======================================

    except Exception as e:

        st.error("❌ Something went wrong.")

        with st.expander("🔧 Technical error"):

            st.code(str(e))
