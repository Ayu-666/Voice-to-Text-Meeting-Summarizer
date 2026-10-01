import json
import uuid
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from groq import Groq

# PDF export is optional: if fpdf2 isn't installed, the rest of the app still works.
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ==========================================
# 1. PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="BBIT Summarizer",
    page_icon="📝",
    layout="wide"
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📝 Smart Meeting & Lecture Summarizer")
st.caption("AI-powered transcription, summaries, decisions and action items.")

st.info(
    "🔒 Audio is sent to Groq for transcription and purged from memory "
    "immediately afterwards. Nothing is stored on disk by this app."
)

# ==========================================
# 2. CONSTANTS
# ==========================================

# Current Groq Whisper model
WHISPER_MODEL = "whisper-large-v3"

# If Groq changes availability, this is the only line you need to change.
LLM_MODEL = "openai/gpt-oss-20b"

MAX_AUDIO_MB = 25
PRIORITIES = ["High", "Medium", "Low"]
TASK_COLUMNS = ["Task Description", "Assigned Person", "Priority", "Deadline"]

# Counter used in widget keys so "Clear all data" can reset uploads/mic too.
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0

rc = st.session_state["reset_counter"]


# ==========================================
# 3. HELPERS
# ==========================================

def parse_deadline(value):
    """Return a date object, or None if the value isn't a valid YYYY-MM-DD."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def fmt_deadline(value):
    d = parse_deadline(value)
    return d.isoformat() if d else "Not specified"


def is_blank(value):
    return value is None or (not isinstance(value, (date, datetime)) and pd.isna(value)) \
        or str(value).strip() == ""


def tasks_to_df(tasks):
    """Convert the AI task list into a clean, editable DataFrame."""
    rows = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        priority = str(t.get("Priority", "Medium")).strip().capitalize()
        if priority not in PRIORITIES:
            priority = "Medium"
        rows.append({
            "Done": False,
            "Task Description": str(t.get("Task Description", "")).strip(),
            "Assigned Person": str(t.get("Assigned Person", "Unassigned")).strip() or "Unassigned",
            "Priority": priority,
            "Deadline": parse_deadline(t.get("Deadline")),
        })
    df = pd.DataFrame(rows, columns=["Done"] + TASK_COLUMNS)
    df["Done"] = df["Done"].astype(bool)
    return df


def df_to_rows(df):
    """Turn the (possibly user-edited) table into clean dicts for exporting."""
    rows = []
    for _, r in df.iterrows():
        if is_blank(r.get("Task Description")):
            continue
        rows.append({
            "done": False if is_blank(r.get("Done")) else bool(r.get("Done")),
            "task": str(r["Task Description"]).strip(),
            "person": "Unassigned" if is_blank(r.get("Assigned Person")) else str(r["Assigned Person"]).strip(),
            "priority": "Medium" if is_blank(r.get("Priority")) else str(r["Priority"]),
            "deadline": parse_deadline(r.get("Deadline")),
        })
    return rows


def build_markdown(summary, decisions, open_items, rows):
    md = "# Meeting Minutes\n\n## Executive Summary\n\n"
    md += "".join(f"- {i}\n" for i in summary) or "No summary available.\n"

    md += "\n## Key Decisions\n\n"
    md += "".join(f"- {i}\n" for i in decisions) or "No decisions detected.\n"

    md += "\n## Open Discussions\n\n"
    md += "".join(f"- {i}\n" for i in open_items) or "No open discussions.\n"

    md += "\n## Action Items\n\n"
    if rows:
        for r in rows:
            box = "x" if r["done"] else " "
            md += (
                f"- [{box}] **{r['task']}**\n"
                f"  - **Assigned:** {r['person']}\n"
                f"  - **Priority:** {r['priority']}\n"
                f"  - **Deadline:** {r['deadline'].isoformat() if r['deadline'] else 'Not specified'}\n\n"
            )
    else:
        md += "No action items detected.\n"
    return md


def _ics_escape(text):
    return (
        str(text).replace("\\", "\\\\").replace(";", "\\;")
        .replace(",", "\\,").replace("\n", "\\n")
    )


def build_ics(rows):
    """Build an .ics calendar file: one all-day event per task that has a date."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//BBIT Summarizer//Action Items//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    count = 0
    for r in rows:
        if not r["deadline"]:
            continue
        count += 1
        start = r["deadline"]
        end = start + timedelta(days=1)  # all-day events end the next day
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uuid.uuid4()}@bbit-summarizer",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{start.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{end.strftime('%Y%m%d')}",
            f"SUMMARY:{_ics_escape(r['task'])}",
            f"DESCRIPTION:{_ics_escape('Assigned to: ' + r['person'] + chr(10) + 'Priority: ' + r['priority'])}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n", count


def _pdf_safe(text):
    # Built-in PDF fonts only support Latin-1, so replace anything else.
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(summary, decisions, open_items, rows):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text, size=13):
        pdf.set_font("Helvetica", "B", size)
        pdf.cell(0, 10, _pdf_safe(text), new_x="LMARGIN", new_y="NEXT")

    def bullets(items, empty_text):
        pdf.set_font("Helvetica", "", 11)
        if not items:
            pdf.multi_cell(0, 6, _pdf_safe(empty_text), new_x="LMARGIN", new_y="NEXT")
        for item in items:
            pdf.multi_cell(0, 6, _pdf_safe(f"- {item}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    heading("Meeting Minutes", 18)
    heading("Executive Summary")
    bullets(summary, "No summary available.")
    heading("Key Decisions")
    bullets(decisions, "No decisions detected.")
    heading("Open Discussions")
    bullets(open_items, "No open discussions.")
    heading("Action Items")

    task_lines = [
        f"[{'x' if r['done'] else ' '}] {r['task']}  |  Owner: {r['person']}  |  "
        f"Priority: {r['priority']}  |  Due: {r['deadline'].isoformat() if r['deadline'] else 'Not specified'}"
        for r in rows
    ]
    bullets(task_lines, "No action items detected.")
    return bytes(pdf.output())


def reset_everything():
    st.session_state.pop("result", None)
    st.session_state["reset_counter"] += 1


# ==========================================
# 4. API KEY
# ==========================================

api_key = st.text_input(
    "Enter your Groq API Key:",
    type="password",
    placeholder="gsk_..."
)

# ==========================================
# 5. INPUT SOURCES
# ==========================================

st.subheader("1. Input Source")

mic_audio = st.audio_input("🎙️ Record from microphone", key=f"mic_{rc}")

file_audio = st.file_uploader(
    f"📁 Or upload an audio file (max {MAX_AUDIO_MB}MB)",
    type=["mp3", "wav", "m4a", "webm"],
    key=f"file_{rc}"
)

raw_text = st.text_area(
    "📝 Or paste a transcript",
    height=180,
    placeholder="Paste your meeting or lecture transcript here...",
    key=f"text_{rc}"
)

# ==========================================
# 6. MAIN BUTTON
# ==========================================

if st.button(
    "✨ Generate Minutes & Tasks",
    type="primary",
    use_container_width=True
):

    if not api_key:
        st.error("❌ Please enter your Groq API key.")
        st.stop()

    audio_src = mic_audio if mic_audio is not None else file_audio

    if audio_src is None and not raw_text.strip():
        st.error("❌ Please provide an audio file, microphone recording, or transcript.")
        st.stop()

    progress = st.progress(0, text="Starting...")

    try:

        client = Groq(api_key=api_key)
        transcript = raw_text.strip()

        # ==================================
        # TRANSCRIPTION
        # ==================================

        if audio_src is not None:

            progress.progress(10, text="📦 Preparing audio...")

            audio_bytes = audio_src.getvalue()

            if not audio_bytes:
                progress.empty()
                st.error("❌ The audio file appears to be empty.")
                st.stop()

            if len(audio_bytes) > MAX_AUDIO_MB * 1024 * 1024:
                progress.empty()
                st.error(f"❌ Audio is larger than {MAX_AUDIO_MB}MB. Please upload a shorter file.")
                st.stop()

            # Use the real filename so Groq detects the right format (mp3, m4a...)
            audio_name = getattr(audio_src, "name", None) or "recording.wav"

            progress.progress(30, text="🎙️ Transcribing audio...")

            transcription = client.audio.transcriptions.create(
                file=(audio_name, audio_bytes),
                model=WHISPER_MODEL,
                response_format="text"
            )

            # Privacy: drop our copy of the audio as soon as it's transcribed
            del audio_bytes

            if isinstance(transcription, str):
                transcript = transcription.strip()
            else:
                transcript = getattr(transcription, "text", str(transcription)).strip()

        if not transcript:
            progress.empty()
            st.error(
                "❌ No speech/transcript was detected. "
                "Try a clearer recording or paste the transcript."
            )
            st.stop()

        # ==================================
        # LLM PROMPT
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
    "topic discussed but not resolved"
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
- "open_discussions" are topics raised but NOT resolved or decided.
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
- If there are no open discussions, return an empty array.
- If there are no tasks, return an empty array.

TRANSCRIPT:

{transcript}
"""

        # ==================================
        # LLM ANALYSIS
        # ==================================

        progress.progress(65, text="🧠 Analyzing transcript...")

        chat_res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=4000
        )

        content = (chat_res.choices[0].message.content or "").strip()

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        # ==================================
        # PARSE JSON
        # ==================================

        try:
            data = json.loads(content)

        except json.JSONDecodeError:

            start = content.find("{")
            end = content.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    "The AI did not return valid JSON.\n\n"
                    f"AI response:\n{content}"
                )

            try:
                data = json.loads(content[start:end + 1])
            except json.JSONDecodeError as json_error:
                raise ValueError(
                    "AI returned malformed JSON.\n\n"
                    f"AI response:\n{content}\n\n"
                    f"JSON error:\n{json_error}"
                )

        if not isinstance(data, dict):
            raise ValueError("AI response was not a JSON object.")

        def as_list(key):
            value = data.get(key, [])
            return value if isinstance(value, list) else []

        # Save everything in session_state so results survive Streamlit reruns
        # (ticking a checkbox or clicking a download button reruns the script).
        st.session_state["result"] = {
            "transcript": transcript,
            "summary": as_list("executive_summary"),
            "decisions": as_list("decisions"),
            "open_items": as_list("open_discussions"),
            "tasks_df": tasks_to_df(as_list("tasks")),
        }

        progress.progress(100, text="✅ Done!")
        progress.empty()

        st.success("✅ Analysis complete!")
        st.toast("Your meeting minutes are ready!", icon="✅")
        st.balloons()

    except Exception as e:

        progress.empty()
        st.error("❌ Something went wrong.")

        with st.expander("🔧 Technical error"):
            st.code(str(e))


# ==========================================
# 7. RESULTS (rendered from session_state)
# ==========================================

res = st.session_state.get("result")

if res:

    summary = res["summary"]
    decisions = res["decisions"]
    open_items = res["open_items"]

    with st.expander("📄 View Transcript"):
        st.write(res["transcript"])

    # ---- Task board first, so metrics and exports see the user's edits ----

    smart_minutes_tab, task_board_tab, exports_tab = st.tabs(
        ["📝 Smart Minutes", "📋 Task Board", "📥 Exports"]
    )

    with task_board_tab:

        st.subheader("📋 Action-Item Matrix")
        st.caption(
            "Edit owners, priorities and dates, tick tasks off, or add a new row "
            "at the bottom of the table. Exports use your edits."
        )

        edited_df = st.data_editor(
            res["tasks_df"],
            key=f"editor_{rc}",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Done": st.column_config.CheckboxColumn("Done", default=False),
                "Task Description": st.column_config.TextColumn("Task Description", required=True),
                "Assigned Person": st.column_config.TextColumn("Assigned Person"),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority", options=PRIORITIES, default="Medium", required=True
                ),
                "Deadline": st.column_config.DateColumn("Deadline", format="YYYY-MM-DD"),
            }
        )

    rows = df_to_rows(edited_df)
    done_count = sum(1 for r in rows if r["done"])

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("📋 Action Items", f"{done_count}/{len(rows)} done")
    metric_col2.metric("✅ Key Decisions", len(decisions))
    metric_col3.metric("💬 Open Discussions", len(open_items))
    metric_col4.metric("📌 Summary Points", len(summary))

    with smart_minutes_tab:

        st.subheader("📌 Executive Summary")
        if summary:
            for item in summary:
                st.markdown(f"• {item}")
        else:
            st.write("No summary available.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Key Decisions")
            if decisions:
                for item in decisions:
                    st.markdown(f"• {item}")
            else:
                st.write("No decisions detected.")

        with col2:
            st.subheader("💬 Open Discussions")
            if open_items:
                for item in open_items:
                    st.markdown(f"• {item}")
            else:
                st.write("No open discussions.")

    with exports_tab:

        st.subheader("📥 Export")

        md_export = build_markdown(summary, decisions, open_items, rows)
        ics_export, ics_count = build_ics(rows)

        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            st.download_button(
                label="📄 Download Markdown",
                data=md_export,
                file_name="meeting_minutes.md",
                mime="text/markdown",
                use_container_width=True
            )

        with ex2:
            if PDF_AVAILABLE:
                try:
                    st.download_button(
                        label="📕 Download PDF",
                        data=build_pdf(summary, decisions, open_items, rows),
                        file_name="meeting_minutes.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pdf_error:
                    st.warning(f"PDF could not be generated: {pdf_error}")
            else:
                st.warning("PDF export needs fpdf2: pip install fpdf2")

        with ex3:
            st.download_button(
                label="📅 Download Calendar (.ics)",
                data=ics_export,
                file_name="action_items.ics",
                mime="text/calendar",
                use_container_width=True,
                disabled=ics_count == 0
            )

        if ics_count == 0:
            st.caption("Calendar export needs at least one task with a deadline date. Add dates in the Task Board tab.")
        else:
            st.caption(f"Calendar file contains {ics_count} event(s), one per task with a deadline.")

    # ---- Privacy: wipe everything on demand ----

    st.divider()
    st.button(
        "🗑️ Clear all data (transcript, results, uploads)",
        on_click=reset_everything,
        use_container_width=True
    )
