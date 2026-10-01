import hashlib
import json
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq


# ==========================================
# Configuration
# ==========================================

st.set_page_config(
    page_title="Smart Meeting Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
SUPPORTED_AUDIO_TYPES = [
    "mp3",
    "wav",
    "m4a",
    "webm",
    "mp4",
    "mpeg",
    "mpga",
    "ogg",
    "flac",
]
PRIORITIES = ["High", "Medium", "Low"]
TASK_COLUMNS = [
    "Done",
    "Task Description",
    "Assigned Person",
    "Priority",
    "Deadline",
]


# ==========================================
# Session state
# ==========================================

def initialize_session_state():
    defaults = {
        "analysis": None,
        "transcript": "",
        "source_fingerprint": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_text(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def safe_bool(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "done", "complete", "completed"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def normalize_task(task):
    task = task if isinstance(task, dict) else {}

    description = safe_text(
        task.get("Task Description", task.get("task", task.get("description", "")))
    )
    assigned = safe_text(
        task.get("Assigned Person", task.get("assignee", task.get("assigned_to", ""))),
        "Unassigned",
    )
    raw_priority = safe_text(task.get("Priority", task.get("priority", "Medium"))).title()
    priority = raw_priority if raw_priority in PRIORITIES else "Medium"
    deadline = safe_text(
        task.get("Deadline", task.get("deadline", task.get("due_date", ""))),
        "Not specified",
    )
    done = safe_bool(task.get("Done", task.get("done", False)))
    status = "Complete" if done else safe_text(task.get("Status", task.get("status", "")), "Open")

    return {
        "Done": done,
        "Task Description": description,
        "Assigned Person": assigned,
        "Priority": priority,
        "Deadline": deadline,
        "Status": status,
        "Evidence": safe_text(task.get("Evidence", task.get("evidence", ""))),
        "Blocker": safe_text(task.get("Blocker", task.get("blocker", ""))),
    }


def as_items(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def normalize_note_items(value):
    notes = []
    for item in as_items(value):
        if isinstance(item, dict):
            text = safe_text(
                item.get(
                    "text",
                    item.get(
                        "decision",
                        item.get("topic", item.get("name", item.get("description", ""))),
                    ),
                )
            )
            evidence = safe_text(item.get("evidence", item.get("Evidence", "")))
        else:
            text = safe_text(item)
            evidence = ""
        if text:
            notes.append({"text": text, "evidence": evidence})
    return notes


def normalize_analysis(raw_data, transcript):
    if not isinstance(raw_data, dict):
        raise ValueError("The AI response must be a JSON object.")

    summary = [
        safe_text(item)
        for item in as_items(raw_data.get("executive_summary", raw_data.get("summary", [])))
        if safe_text(item)
    ]
    tasks = []
    for raw_task in as_items(raw_data.get("tasks", [])):
        normalized = normalize_task(raw_task)
        if normalized["Task Description"]:
            tasks.append(normalized)

    title = safe_text(
        raw_data.get("meeting_title", raw_data.get("title", "")),
        "Meeting / lecture notes",
    )
    content_type = safe_text(
        raw_data.get("content_type", raw_data.get("meeting_type", "")),
        "Not specified",
    ).title()
    if content_type.lower() not in {"meeting", "lecture", "mixed", "not specified"}:
        content_type = "Not specified"

    return {
        "meeting_title": title,
        "content_type": content_type,
        "date": safe_text(raw_data.get("date", ""), "Not specified"),
        "participants": normalize_note_items(raw_data.get("participants", [])),
        "topics": normalize_note_items(raw_data.get("topics", [])),
        "executive_summary": summary,
        "decisions": normalize_note_items(raw_data.get("decisions", [])),
        "tasks": tasks,
        "blockers": normalize_note_items(raw_data.get("blockers", [])),
        "follow_ups": normalize_note_items(raw_data.get("follow_ups", [])),
        "open_questions": normalize_note_items(raw_data.get("open_questions", [])),
        "transcript": transcript,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def sync_task_editor():
    """Copy data-editor changes into durable session state before rerendering."""
    current_analysis = st.session_state.get("analysis")
    if not isinstance(current_analysis, dict):
        return

    edited_value = st.session_state.get("task_editor")
    original_tasks = current_analysis.get("tasks", [])
    is_delta = isinstance(edited_value, dict) and any(
        key in edited_value for key in ("edited_rows", "added_rows", "deleted_rows")
    )

    if isinstance(edited_value, pd.DataFrame):
        records = edited_value.to_dict(orient="records")
    elif isinstance(edited_value, list):
        records = edited_value
    elif is_delta:
        # Streamlit versions may store a data-editor delta in session state instead of
        # the complete edited DataFrame.
        records = []
        for task in original_tasks:
            normalized = normalize_task(task)
            row = {column: normalized[column] for column in TASK_COLUMNS}
            row.update({
                "Evidence": normalized["Evidence"],
                "Blocker": normalized["Blocker"],
            })
            records.append(row)

        edited_rows = edited_value.get("edited_rows", {})
        if isinstance(edited_rows, dict):
            for row_index, changes in edited_rows.items():
                try:
                    row_index = int(row_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= row_index < len(records) and isinstance(changes, dict):
                    records[row_index].update(changes)

        deleted_rows = set()
        for row_index in edited_value.get("deleted_rows", []):
            try:
                deleted_rows.add(int(row_index))
            except (TypeError, ValueError):
                continue
        records = [
            row for index, row in enumerate(records)
            if index not in deleted_rows
        ]

        added_rows = edited_value.get("added_rows", [])
        if isinstance(added_rows, list):
            for row in added_rows:
                if isinstance(row, dict):
                    new_task = dict(row)
                    new_task.setdefault("Evidence", "")
                    new_task.setdefault("Blocker", "")
                    records.append(new_task)
    else:
        return

    updated_tasks = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            continue

        original = original_tasks[index] if index < len(original_tasks) else {}
        task = normalize_task(
            {
                **row,
                "Evidence": row.get("Evidence", original.get("Evidence", "")),
                "Blocker": row.get("Blocker", original.get("Blocker", "")),
            }
        )
        if task["Task Description"]:
            task["Status"] = "Complete" if task["Done"] else "Open"
            updated_tasks.append(task)

    current_analysis["tasks"] = updated_tasks
    st.session_state["analysis"] = current_analysis
    if is_delta:
        # The canonical task list now contains these edits; clearing the delta avoids
        # replaying added/deleted rows against the updated list on a later rerun.
        st.session_state["task_editor"] = {
            "edited_rows": {},
            "added_rows": [],
            "deleted_rows": [],
        }


initialize_session_state()


# ==========================================
# Groq and response helpers
# ==========================================

def get_saved_api_key():
    try:
        return safe_text(st.secrets.get("GROQ_API_KEY", ""))
    except Exception:
        return ""


def audio_filename(audio_source):
    original_name = safe_text(getattr(audio_source, "name", ""), "recording.wav")
    original_name = os.path.basename(original_name)
    original_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)

    if "." not in original_name:
        mime_type = safe_text(getattr(audio_source, "type", "")).lower()
        extension_by_mime = {
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mp4": "m4a",
            "audio/x-m4a": "m4a",
            "audio/webm": "webm",
            "audio/ogg": "ogg",
            "audio/flac": "flac",
        }
        extension = extension_by_mime.get(mime_type, "wav")
        original_name = f"recording.{extension}"

    return original_name


def source_hash(kind, content):
    return hashlib.sha256(kind.encode("utf-8") + b"\0" + content).hexdigest()


def transcribe_audio(client, audio_source, audio_bytes):
    if not audio_bytes:
        raise ValueError("The selected audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            "This audio file is larger than Groq's 25 MB transcription limit. "
            "Use a smaller or compressed file."
        )

    filename = audio_filename(audio_source)
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=WHISPER_MODEL,
        response_format="text",
    )

    if isinstance(transcription, str):
        transcript = transcription.strip()
    else:
        transcript = safe_text(getattr(transcription, "text", transcription))

    if not transcript:
        raise ValueError(
            "No speech was detected in the audio. Try a clearer recording or paste a transcript."
        )
    return transcript


def build_prompt(transcript):
    return f"""
You are an evidence-grounded assistant for meetings and lectures. Analyze the transcript and return
one valid JSON object only. Do not use Markdown fences or add facts from outside the transcript.

Rules:
- Classify the content as Meeting, Lecture, Mixed, or Not specified.
- Use only names, dates, decisions, commitments, and tasks that the speaker explicitly states or
  that are unambiguously agreed in the transcript.
- A suggestion, possibility, question, or unaccepted proposal is not a decision or a task.
- Never invent participants, assignees, priorities, deadlines, or task status.
- Only include tasks that have a concrete action. Use "Unassigned" when no owner is stated and
  "Not specified" when no deadline is stated.
- Use High, Medium, or Low for priority only when supported by urgency/importance in the transcript;
  otherwise use Medium.
- Keep summaries concise and informative. Return 3–5 summary bullets when the transcript supports it.
- Add brief exact or faithful evidence snippets for each decision and task. Leave evidence empty if
  there is no concise supporting excerpt; do not fabricate a quote.
- If something is absent or uncertain, use an empty array or "Not specified", as appropriate.

Return exactly this shape:
{{
  "meeting_title": "short title or Meeting / lecture notes",
  "content_type": "Meeting | Lecture | Mixed | Not specified",
  "date": "explicit date or Not specified",
  "participants": ["names explicitly mentioned as participants"],
  "topics": ["topic"],
  "executive_summary": ["concise point"],
  "decisions": [{{"text": "decision", "evidence": "supporting excerpt"}}],
  "tasks": [
    {{
      "Task Description": "specific action",
      "Assigned Person": "explicit owner or Unassigned",
      "Priority": "High | Medium | Low",
      "Deadline": "explicit date/deadline or Not specified",
      "Evidence": "supporting excerpt",
      "Blocker": "stated blocker or empty string"
    }}
  ],
  "blockers": [{{"text": "stated blocker", "evidence": "supporting excerpt"}}],
  "follow_ups": [{{"text": "explicit follow-up", "evidence": "supporting excerpt"}}],
  "open_questions": ["question that remains unresolved"]
}}

TRANSCRIPT:
{transcript}
"""


def extract_json_object(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The AI returned an empty response.")

    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("The AI response did not contain a JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        character = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]

    raise ValueError("The AI response contained incomplete or malformed JSON.")


def parse_analysis(content, transcript):
    json_text = extract_json_object(content)
    try:
        raw_data = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The AI response was malformed JSON: {error.msg}.") from None
    return normalize_analysis(raw_data, transcript)


def friendly_api_error(error, phase):
    """Return a safe user-facing error without exposing provider payloads or credentials."""
    detail = str(error).lower()

    if any(token in detail for token in ("401", "invalid_api_key", "authentication", "unauthorized")):
        return "Groq rejected the API key. Check the key and try again."
    if any(token in detail for token in ("429", "rate_limit", "rate limit", "too many requests")):
        return "Groq rate limit reached. Wait a moment, then try again."
    if any(token in detail for token in ("timeout", "timed out", "connection", "network")):
        return f"Network trouble interrupted {phase}. Check your connection and retry."
    if any(token in detail for token in ("413", "payload too large", "file too large")):
        return "The audio is too large for transcription. Use a smaller or compressed file."
    if any(token in detail for token in ("unsupported", "invalid file", "format")):
        return "Groq could not read this audio format. Try MP3, WAV, M4A, WEBM, OGG, or FLAC."
    return f"{phase.capitalize()} failed. Check your Groq access and connection, then try again."


# ==========================================
# Export helpers
# ==========================================

def task_dataframe(tasks):
    rows = []
    for task in tasks:
        normalized = normalize_task(task)
        rows.append({column: normalized[column] for column in TASK_COLUMNS})
    return pd.DataFrame(rows, columns=TASK_COLUMNS)


def render_note_section(title, notes):
    if not notes:
        return
    st.markdown(f"**{title}**")
    for note in notes:
        st.markdown(f"- {note['text']}")
        if note.get("evidence"):
            st.caption(f'  Evidence: “{note["evidence"]}”')


def markdown_export(analysis):
    lines = [
        f"# {analysis.get('meeting_title', 'Meeting / lecture notes')}",
        "",
        f"- **Type:** {analysis.get('content_type', 'Not specified')}",
        f"- **Date:** {analysis.get('date', 'Not specified')}",
        f"- **Generated:** {analysis.get('generated_at', 'Not specified')}",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(f"- {item}" for item in analysis.get("executive_summary", []))

    for section_title, key in (
        ("Topics", "topics"),
        ("Participants", "participants"),
        ("Decisions", "decisions"),
        ("Blockers", "blockers"),
        ("Follow-ups", "follow_ups"),
        ("Open Questions", "open_questions"),
    ):
        notes = analysis.get(key, [])
        if notes:
            lines.extend(["", f"## {section_title}", ""])
            for note in notes:
                lines.append(f"- {note['text']}")
                if note.get("evidence"):
                    lines.append(f'  - Evidence: “{note["evidence"]}”')

    lines.extend(["", "## Action Items", ""])
    tasks = analysis.get("tasks", [])
    if not tasks:
        lines.append("No action items detected.")
    else:
        for task in tasks:
            status = "Complete" if task.get("Done") else "Open"
            lines.extend(
                [
                    f"- **{task.get('Task Description', 'Not specified')}**",
                    f"  - Assigned: {task.get('Assigned Person', 'Unassigned')}",
                    f"  - Priority: {task.get('Priority', 'Medium')}",
                    f"  - Deadline: {task.get('Deadline', 'Not specified')}",
                    f"  - Status: {status}",
                ]
            )
            if task.get("Evidence"):
                lines.append(f'  - Evidence: “{task["Evidence"]}”')

    return "\n".join(lines).strip() + "\n"


# ==========================================
# Page layout
# ==========================================

st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1400px;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(99, 102, 241, .10), rgba(14, 165, 233, .06));
            border: 1px solid rgba(99, 102, 241, .16);
            padding: 1rem 1.1rem;
            border-radius: 14px;
        }
        div[data-testid="stTabs"] button {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 Smart Meeting & Lecture Summarizer")
st.caption(
    "Turn a recording or transcript into grounded minutes, decisions, and an editable action board."
)

with st.sidebar:
    st.header("⚙️ Settings")
    saved_api_key = get_saved_api_key()
    manual_api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="A key in Streamlit Secrets named GROQ_API_KEY is used when this field is blank.",
        key="manual_groq_api_key",
    )
    api_key = manual_api_key.strip() or saved_api_key
    if saved_api_key and not manual_api_key.strip():
        st.caption("Using GROQ_API_KEY from Streamlit Secrets.")
    st.caption("Audio is sent to Groq for transcription and analysis.")
    st.divider()
    if st.button("🧹 Clear current results", use_container_width=True):
        st.session_state["analysis"] = None
        st.session_state["transcript"] = ""
        st.session_state["source_fingerprint"] = ""
        st.session_state.pop("task_editor", None)
        st.toast("Current results cleared.", icon="🧹")

st.subheader("1. Add a recording or transcript")
input_col1, input_col2 = st.columns(2)

with input_col1:
    mic_audio = st.audio_input("🎙️ Record from microphone", key="mic_audio_input")
    uploaded_audio = st.file_uploader(
        "📁 Or upload an audio file",
        type=SUPPORTED_AUDIO_TYPES,
        help="Supported formats include MP3, WAV, M4A, WEBM, MP4, OGG, and FLAC.",
        key="audio_file_uploader",
    )

with input_col2:
    raw_text = st.text_area(
        "📝 Or paste a transcript",
        height=220,
        placeholder="Paste meeting notes, a lecture transcript, or a transcript from another tool…",
        key="raw_transcript_input",
    )
    st.caption("If you provide audio and text together, the audio takes priority.")

generate = st.button(
    "✨ Generate minutes & tasks",
    type="primary",
    use_container_width=True,
    key="generate_analysis",
)


# ==========================================
# Generate / update analysis
# ==========================================

if generate:
    audio_source = mic_audio if mic_audio is not None else uploaded_audio
    transcript_input = raw_text.strip()

    if not api_key:
        st.error("Add a Groq API key in the sidebar or configure GROQ_API_KEY in Streamlit Secrets.")
    elif audio_source is None and not transcript_input:
        st.error("Record or upload audio, or paste a transcript before generating minutes.")
    else:
        try:
            client = Groq(api_key=api_key)

            if audio_source is not None:
                audio_bytes = audio_source.getvalue()
                fingerprint = source_hash("audio", audio_bytes)
                cached_transcript = st.session_state.get("transcript", "")
                use_cached = (
                    bool(cached_transcript)
                    and st.session_state.get("source_fingerprint") == fingerprint
                )

                if use_cached:
                    transcript = cached_transcript
                    st.info("Reusing the transcript already created for this audio.")
                else:
                    with st.spinner("🎙️ Transcribing audio with Groq Whisper…"):
                        transcript = transcribe_audio(client, audio_source, audio_bytes)
                    st.session_state["transcript"] = transcript
                    st.session_state["source_fingerprint"] = fingerprint
            else:
                transcript = transcript_input
                fingerprint = source_hash("text", transcript.encode("utf-8"))
                st.session_state["transcript"] = transcript
                st.session_state["source_fingerprint"] = fingerprint

            if not transcript.strip():
                st.error("No transcript was available to analyze.")
            else:
                with st.spinner("🧠 Analyzing the transcript…"):
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You extract concise, evidence-grounded meeting and lecture "
                                    "information. Return only the requested JSON object."
                                ),
                            },
                            {"role": "user", "content": build_prompt(transcript)},
                        ],
                        temperature=0,
                        max_tokens=3000,
                    )

                content = completion.choices[0].message.content or ""
                analysis = parse_analysis(content, transcript)
                st.session_state["analysis"] = analysis
                st.session_state.pop("task_editor", None)
                st.success("✅ Analysis complete. Your results are saved for this session.")
                st.toast("Meeting minutes and tasks are ready!", icon="✅")
                st.balloons()

        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            phase = "transcription" if audio_source is not None and not st.session_state.get("transcript") else "analysis"
            st.error(friendly_api_error(error, phase))


# ==========================================
# Persistent results
# ==========================================

analysis = st.session_state.get("analysis")

if isinstance(analysis, dict):
    tasks = analysis.get("tasks", [])
    decisions = analysis.get("decisions", [])
    completed_count = sum(1 for task in tasks if safe_bool(task.get("Done")))
    pending_count = max(len(tasks) - completed_count, 0)

    st.divider()
    st.subheader(analysis.get("meeting_title", "Meeting / lecture notes"))
    st.caption(
        f"{analysis.get('content_type', 'Not specified')} · "
        f"Date: {analysis.get('date', 'Not specified')} · "
        f"Generated: {analysis.get('generated_at', 'Not specified')}"
    )
    st.info("AI-generated details can be incomplete; verify decisions and action items before sharing.")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("📋 Action items", len(tasks))
    metric2.metric("✅ Decisions", len(decisions))
    metric3.metric("☑️ Completed tasks", completed_count, delta=f"{pending_count} remaining")

    minutes_tab, tasks_tab, exports_tab = st.tabs(
        ["📝 Smart Minutes", "📋 Task Board", "📥 Exports"]
    )

    with minutes_tab:
        summary_col, details_col = st.columns([1.25, 0.75], gap="large")

        with summary_col:
            st.markdown("### Executive summary")
            if analysis.get("executive_summary"):
                for item in analysis["executive_summary"]:
                    st.markdown(f"- {item}")
            else:
                st.write("No summary points were detected.")

            render_note_section("Key decisions", analysis.get("decisions", []))

        with details_col:
            render_note_section("Topics", analysis.get("topics", []))
            render_note_section("Participants", analysis.get("participants", []))
            render_note_section("Blockers", analysis.get("blockers", []))
            render_note_section("Follow-ups", analysis.get("follow_ups", []))
            render_note_section("Open questions", analysis.get("open_questions", []))

        with st.expander("📄 View transcript"):
            st.text(analysis.get("transcript", "No transcript saved."))

    with tasks_tab:
        st.markdown("### Action-item board")
        st.caption(
            "Edit cells, add rows, or mark tasks complete. Changes are saved in this Streamlit session."
        )

        task_df = task_dataframe(tasks)
        st.data_editor(
            task_df,
            key="task_editor",
            on_change=sync_task_editor,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_order=TASK_COLUMNS,
            column_config={
                "Done": st.column_config.CheckboxColumn(
                    "Done",
                    help="Mark this action item complete.",
                ),
                "Task Description": st.column_config.TextColumn(
                    "Task",
                    help="Describe the specific action.",
                    required=True,
                ),
                "Assigned Person": st.column_config.TextColumn("Assigned to"),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=PRIORITIES,
                    required=True,
                ),
                "Deadline": st.column_config.TextColumn(
                    "Deadline",
                    help="Date or deadline stated in the transcript.",
                ),
            },
        )

        live_tasks = analysis.get("tasks", [])
        evidence_tasks = [task for task in live_tasks if task.get("Evidence")]
        if evidence_tasks:
            with st.expander("🔎 Task evidence from the transcript"):
                for task in evidence_tasks:
                    st.markdown(f"**{task['Task Description']}**")
                    st.caption(f'“{task["Evidence"]}”')

    with exports_tab:
        st.markdown("### Download your results")
        st.caption("Exports include the task edits currently saved in this session.")

        current_analysis = st.session_state.get("analysis", analysis)
        current_tasks = current_analysis.get("tasks", [])
        export_df = task_dataframe(current_tasks)
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        json_data = json.dumps(current_analysis, ensure_ascii=False, indent=2)
        md_data = markdown_export(current_analysis)

        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            st.download_button(
                "⬇️ Download task CSV",
                data=csv_data,
                file_name="meeting_tasks.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_tasks_csv",
            )
        with export_col2:
            st.download_button(
                "⬇️ Download full JSON",
                data=json_data,
                file_name="meeting_analysis.json",
                mime="application/json",
                use_container_width=True,
                key="download_analysis_json",
            )
        with export_col3:
            st.download_button(
                "⬇️ Download Markdown minutes",
                data=md_data,
                file_name="meeting_minutes.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_minutes_markdown",
            )

        st.caption("The JSON export contains the transcript along with analysis and task-board data.")

elif st.session_state.get("transcript"):
    st.info(
        "A transcript is saved for this session, but analysis is not complete. "
        "Correct any API issue and select Generate again; the same audio transcript will be reused."
    )

st.caption("Powered by Groq Whisper and Groq language models · Results remain in this browser session.")
