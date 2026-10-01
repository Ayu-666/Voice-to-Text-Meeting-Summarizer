import json
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from fpdf import FPDF  # requirements.txt: fpdf2
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
# 4b. EXPORT HELPERS (PDF + ICS)
# ==========================================
# NOTE: Streamlit runs top-to-bottom, so these must be defined
# BEFORE the button block below that calls them.

def generate_ics(df):
    """Build an .ics calendar (all-day events) from tasks that have a YYYY-MM-DD deadline."""

    def esc(text):
        return (
            str(text)
            .replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\r\n", "\\n")
            .replace("\n", "\\n")
        )

    def fold(line, limit=70):
        parts = [line[i:i + limit] for i in range(0, len(line), limit)]
        return "\r\n ".join(parts)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//BBIT Summarizer//Meeting Tasks//EN",
        "CALSCALE:GREGORIAN",
    ]

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            try:
                due = datetime.strptime(str(row.get("Deadline", "")).strip(), "%Y-%m-%d")
            except ValueError:
                continue  # "Not specified" or malformed date

            task = row.get("Task Description", "Task")
            person = row.get("Assigned Person", "Unassigned")
            priority = row.get("Priority", "Medium")

            lines += [
                "BEGIN:VEVENT",
                f"UID:{uuid.uuid4()}@bbit-summarizer",
                f"DTSTAMP:{stamp}",
                f"DTSTART;VALUE=DATE:{due.strftime('%Y%m%d')}",
                f"DTEND;VALUE=DATE:{(due + timedelta(days=1)).strftime('%Y%m%d')}",
                "SUMMARY:" + esc("[" + str(priority) + "] " + str(task)),
                "DESCRIPTION:" + esc("Assigned to: " + str(person) + "\nPriority: " + str(priority)),
                "END:VEVENT",
            ]

    lines.append("END:VCALENDAR")

    return "\r\n".join(fold(line) for line in lines) + "\r\n"


def generate_pdf(data, df):
    """Build a PDF of the minutes. Returns bytes."""

    def safe(value):
        # Built-in PDF fonts are Latin-1 only; normalise common Unicode.
        text = str(value)
        for old, new in {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2022": "-", "\u2026": "...",
        }.items():
            text = text.replace(old, new)
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Meeting Minutes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        new_x="LMARGIN", new_y="NEXT"
    )
    pdf.ln(4)

    def section(title, items):
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, safe(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        if items:
            for item in items:
                pdf.multi_cell(0, 6, safe(f"- {item}"), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.multi_cell(0, 6, "None recorded.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    section("Executive Summary", data.get("executive_summary", []))
    section("Key Decisions", data.get("decisions", []))
    section("Open Discussions", data.get("open_discussions", []))

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Action Items", new_x="LMARGIN", new_y="NEXT")

    if df is not None and not df.empty:
        pdf.set_font("Helvetica", "", 9)
        with pdf.table(col_widths=(14, 84, 36, 24, 32), text_align="LEFT") as table:
            table.row(["Done", "Task", "Assigned", "Priority", "Deadline"])
            for _, row in df.iterrows():
                table.row([
                    "Yes" if bool(row.get("Done", False)) else "No",
                    safe(row.get("Task Description", "")),
                    safe(row.get("Assigned Person", "")),
                    safe(row.get("Priority", "")),
                    safe(row.get("Deadline", "")),
                ])
    else:
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, "No action items detected.", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


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

    progress = st.progress(0, text="Starting...")

    try:

        client = Groq(api_key=api_key)

        transcript = raw_text.strip()

        # ==================================
        # 6. TRANSCRIPTION
        # ==================================

        if audio_src is not None:

            progress.progress(10, text="🎙️ Transcribing audio... (10%)")


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

        else:
            progress.progress(10, text="📝 Preparing transcript... (10%)")

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
  "open_discussions": [
    "unresolved topic 1"
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
- "open_discussions" are topics that were raised but NOT resolved, or that need follow-up. Do not repeat decisions here.
- If there are no open discussions, return an empty array.
- If there are no tasks, return an empty array.

TRANSCRIPT:

{transcript}
"""

        # ==================================
        # 9. LLM ANALYSIS
        # ==================================

        progress.progress(60, text="🧠 Analyzing transcript... (60%)")

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
        open_discussions = data.get("open_discussions", [])
        tasks = data.get("tasks", [])

        if not isinstance(summary, list):
            summary = []

        if not isinstance(decisions, list):
            decisions = []

        if not isinstance(open_discussions, list):
            open_discussions = []

        if not isinstance(tasks, list):
            tasks = []

        # ==================================
        # 13. SUCCESS
        # ==================================

        progress.progress(100, text="✅ Done! (100%)")

        st.success("✅ Analysis complete!")

        # ==================================
        # 14. SUMMARY + DECISIONS
        # ==================================

        col1, col2, col3 = st.columns(3)

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

        with col3:

            st.subheader("💬 Open Discussions")

            if open_discussions:

                for item in open_discussions:
                    st.markdown(f"• {item}")

            else:
                st.write("No open discussions detected.")

        # ==================================
        # 15. ACTION ITEMS
        # ==================================

        st.subheader("📋 Action-Item Matrix")

        # Default so exports always work, even with zero tasks
        edited_df = pd.DataFrame()

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

            with st.container(border=True):

                st.markdown("#### 🗂️ InteractiveTaskBoard")
                st.caption("Tick tasks off or edit any cell directly.")

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

        md_export += "\n## Open Discussions\n\n"

        for item in open_discussions:
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

        export_data = {
            "executive_summary": summary,
            "decisions": decisions,
            "open_discussions": open_discussions,
        }

        # PDF is built separately so a PDF problem never hides the other exports
        try:
            pdf_bytes = generate_pdf(export_data, edited_df)
        except Exception:
            pdf_bytes = None

        ics_text = generate_ics(edited_df)

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                label="📥 Download Markdown",
                data=md_export,
                file_name="meeting_minutes.md",
                mime="text/markdown",
                use_container_width=True
            )

        with dl2:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes or b"",
                file_name="meeting_minutes.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=pdf_bytes is None
            )

        with dl3:
            st.download_button(
                label="📅 Download Calendar (.ics)",
                data=ics_text,
                file_name="meeting_tasks.ics",
                mime="text/calendar",
                use_container_width=True,
                disabled="BEGIN:VEVENT" not in ics_text,
                help="Adds tasks that have a deadline date to your calendar."
            )

    # ======================================
    # 17. ERROR HANDLING
    # ======================================

    except Exception as e:

        progress.empty()

        st.error("❌ Something went wrong.")

        with st.expander("🔧 Technical error"):

            st.code(str(e))
