"""
MeetMind - AI Meeting & Lecture Summarizer with Action-Item Tracker
BBIT Hackathon 2026 | Problem AI-03 (Groq-Only Edition)
"""
import html
import json
import os
import re
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MeetMind | AI Summarizer", page_icon="🎙️", layout="wide")

MAX_MB = 25
GROQ_MODEL_AUDIO = "whisper-large-v3-turbo"
GROQ_MODEL_TEXT = "llama-3.3-70b-versatile"
PRIORITIES = ["High", "Medium", "Low"]
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
STATUSES = ["To Do", "In Progress"]
TASK_COLS = ["Done", "Task", "Assignee", "Priority", "Status", "Deadline", "Implied Deadline", "Evidence"]
DUE_SOON_DAYS = 3

SAMPLE_MEETING = """Riya: Good morning everyone. Let's start our weekly sync for the hackathon project. We need to lock the plan for the final demo.
Arjun: Backend update: the API endpoints are done, but I still have to connect the database and add error handling.
Meera: On the design side, the main dashboard is ready. I haven't started the export screens yet.
Riya: First decision. Are we using Streamlit for the frontend?
Meera: Yes, Streamlit. It's faster for us and we can deploy it for free.
Arjun: Agreed. Let's finalize Streamlit then.
Riya: Done, Streamlit is final. Second, we'll use Groq Whisper for transcription.
Riya: Now action items. Arjun, please finish the database integration by this Friday. It's our top priority, the demo depends on it.
Arjun: Got it, I'll have it done by Friday.
Meera: I'll finish the export screens and share the final mockups with the team by next Monday.
Sam: I'll write the README and the docs folder and submit everything by October 10. Medium priority, but it's a hard deadline.
Riya: Sam, could you also book the seminar hall for a dry run? No rush, just sometime this month.
Sam: Sure, I'll do that.
Arjun: One more thing. Should we add PDF export? I'm not sure we have the time.
Riya: Let's keep that open and discuss it tomorrow. We also haven't decided who will present the demo to the judges.
Meera: Let's pick the presenter at the next meeting.
Riya: Okay, that's everything. Thanks, all."""

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {--mm-primary:#4f46e5; --mm-border:rgba(128,128,128,.28); --mm-soft:rgba(128,128,128,.07);}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px;}
#MainMenu, footer {visibility: hidden;}
.mm-header {display:flex; align-items:center; gap:14px; margin-bottom:.3rem;}
.mm-logo {font-size:2.2rem; line-height:1;}
.mm-title {font-size:2rem; font-weight:800; letter-spacing:-.6px; color:var(--mm-primary); line-height:1.1;}
.mm-sub {font-size:1.02rem; font-weight:600; opacity:.85;}
.mm-tag {font-size:.92rem; opacity:.6; font-style:italic;}
.pipeline {display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:.9rem 0 1.1rem 0;}
.pipeline .step {border:1px solid var(--mm-border); border-radius:999px; padding:5px 14px; font-size:.85rem; font-weight:600; background:var(--mm-soft);}
.pipeline .arrow {opacity:.45; font-weight:700;}
.mm-card {border:1px solid var(--mm-border); border-radius:14px; padding:1rem 1.25rem; background:var(--mm-soft); box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:.8rem;}
.mm-headline {border-left:5px solid var(--mm-primary);}
.mm-headline .lbl {font-size:.75rem; font-weight:700; letter-spacing:.8px; opacity:.6; text-transform:uppercase;}
.mm-headline .txt {font-size:1.2rem; font-weight:650; margin-top:.2rem; line-height:1.4;}
.mm-headline .ttl {font-size:.92rem; opacity:.7; margin-top:.4rem;}
.mm-note {font-size:.82rem; opacity:.65; margin:.1rem 0 1rem 0;}
.mm-chips {display:flex; flex-wrap:wrap; gap:8px; margin:.4rem 0 .6rem 0;}
.mm-chip {border:1px solid var(--mm-border); border-radius:8px; padding:3px 10px; font-size:.78rem; opacity:.85;}
.mm-section {font-size:1.05rem; font-weight:700; margin:.2rem 0 .5rem 0;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:700; color:#fff; margin-right:6px; white-space:nowrap;}
.b-High{background:#dc2626;} .b-Medium{background:#d97706;} .b-Low{background:#059669;}
.s-todo{background:#64748b;} .s-prog{background:#2563eb;} .s-done{background:#059669;}
.d-overdue{background:#dc2626;} .d-soon{background:#d97706;} .d-upcoming{background:#4f46e5;} .d-nodate{background:#94a3b8;}
.task {border:1px solid var(--mm-border); border-left:5px solid var(--mm-primary); border-radius:12px; padding:.75rem 1rem; margin-bottom:.6rem; background:var(--mm-soft);}
.task.done {opacity:.6;} .task.done .t {text-decoration:line-through;}
.task .t {font-weight:650; font-size:1rem; margin-bottom:.4rem;}
.task .m {font-size:.84rem; opacity:.85; margin-top:.3rem;}
.task .ph {font-style:italic; opacity:.75;}
.kcol {font-weight:700; padding:.35rem .1rem; border-bottom:3px solid; margin-bottom:.7rem;}
.kcard {border:1px solid var(--mm-border); border-radius:12px; padding:.7rem .85rem; margin-bottom:.6rem; background:var(--mm-soft); box-shadow:0 1px 2px rgba(0,0,0,.05);}
.kcard .t {font-weight:600; margin-bottom:.4rem;}
.kcard .m {font-size:.8rem; opacity:.85; margin-top:.3rem;}
.tbox {max-height:440px; overflow-y:auto; border:1px solid var(--mm-border); border-radius:12px; padding:1rem 1.3rem; background:var(--mm-soft); line-height:1.75; font-size:.97rem;}
.tbox p {margin:0 0 .9rem 0;}
.tbox .spk {font-weight:700; margin-right:6px;}
.exp {border:1px solid var(--mm-border); border-radius:14px; padding:1rem 1.1rem; background:var(--mm-soft); margin-bottom:.6rem; min-height:118px;}
.exp .h {font-weight:700; font-size:1.02rem;}
.exp .d {font-size:.86rem; opacity:.75; margin-top:.25rem;}
.mm-footer {text-align:center; font-size:.8rem; opacity:.5; margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--mm-border);}
div[data-testid="stMetric"] {border:1px solid var(--mm-border); border-radius:14px; padding:.7rem 1rem; background:var(--mm-soft); box-shadow:0 1px 3px rgba(0,0,0,.05);}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Session State & Helpers
# ----------------------------------------------------------------------------
def empty_tasks_df():
    df = pd.DataFrame(columns=TASK_COLS)
    df["Done"] = df["Done"].astype(bool)
    df["Deadline"] = pd.to_datetime(df["Deadline"])
    return df

defaults = {"transcript": "", "minutes": None, "source": "text", "tasks": empty_tasks_df(), 
            "tasks_base": empty_tasks_df(), "tasks_ver": 0, "upload_n": 0, "rec_n": 0, 
            "notice": "", "paste_text": "", "input_mode": "upload", "kind": "Meeting"}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

class UserError(Exception): pass

def get_key():
    val = os.environ.get("GROQ_API_KEY")
    if val: return val
    try: return st.secrets["GROQ_API_KEY"]
    except: return st.session_state.get("key_GROQ_API_KEY", "")

def set_tasks(df):
    st.session_state.tasks = df
    st.session_state.tasks_base = df.copy()
    st.session_state.tasks_ver += 1

def purge_all():
    st.session_state.transcript = ""
    st.session_state.minutes = None
    st.session_state.paste_text = ""
    st.session_state.upload_n += 1
    st.session_state.rec_n += 1
    set_tasks(empty_tasks_df())

def load_sample():
    st.session_state.paste_text = SAMPLE_MEETING
    st.session_state.input_mode = "paste"

def add_blank_task():
    new = pd.DataFrame([{"Done": False, "Task": "New task", "Assignee": "Unassigned", "Priority": "Medium",
                         "Status": "To Do", "Deadline": pd.NaT, "Implied Deadline": "", "Evidence": ""}], columns=TASK_COLS)
    cur = st.session_state.tasks.copy()
    cur.loc[cur["Status"] == "Done", "Status"] = "To Do"
    merged = new if cur.empty else pd.concat([cur, new], ignore_index=True)
    merged["Deadline"] = pd.to_datetime(merged["Deadline"], errors="coerce")
    merged["Done"] = merged["Done"].astype(bool)
    set_tasks(merged)

# ----------------------------------------------------------------------------
# AI Core (Groq Only)
# ----------------------------------------------------------------------------
def transcribe_audio(client, audio_bytes, filename):
    try:
        res = client.audio.transcriptions.create(file=(filename, audio_bytes), model=GROQ_MODEL_AUDIO)
        return res.text.strip()
    except Exception as e:
        raise UserError(f"Transcription failed: {str(e)}")

def build_prompt(transcript, attendees):
    today = date.today()
    schema = """{
  "title": "short descriptive title of the meeting",
  "attendees": ["names of people who clearly took part"],
  "headline": "ONE sentence: the single most important outcome of the meeting",
  "summary": ["exactly 3 concise bullets forming the executive summary"],
  "decisions": ["decisions that were clearly agreed/ratified"],
  "discussion_points": ["main topics that were discussed"],
  "open_discussions": ["issues raised but NOT resolved, or questions left open"],
  "action_items": [{"task": "...", "assignee": "...", "priority": "High|Medium|Low", "deadline_text": "...", "deadline_date": "YYYY-MM-DD or null", "evidence": "exact words from the transcript"}]
}"""
    att = f"\nKnown attendees (optional hint): {attendees}" if attendees else ""
    return f"""You are an expert minute-taker for meetings. Analyse the transcript and return ONLY valid JSON matching this schema:
{schema}

GROUND RULES
- Use ONLY information present in the transcript. Do not invent facts or names.
- If a section has nothing to report, return an empty list.
- Today's date is {today.isoformat()} ({today.strftime('%A')}).
- "deadline_date": format as YYYY-MM-DD. For vague wording ("soon", "later"), use null.
- Priority: High / Medium / Low. {att}

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\""""

def extract_minutes(client, transcript, attendees):
    prompt = build_prompt(transcript, attendees)
    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(res.choices[0].message.content)
        
        # Normalization
        items = []
        for it in data.get("action_items", []):
            if not str(it.get("task", "")).strip(): continue
            pr = str(it.get("priority", "Medium")).strip().title()
            dl_date = it.get("deadline_date")
            try: dl_date = datetime.strptime(str(dl_date).strip()[:10], "%Y-%m-%d").date().isoformat()
            except: dl_date = None
            
            items.append({
                "task": str(it["task"]).strip(),
                "assignee": str(it.get("assignee", "Unassigned")).strip() or "Unassigned",
                "priority": pr if pr in PRIORITIES else "Medium",
                "deadline_text": str(it.get("deadline_text", "")).strip(),
                "deadline_date": dl_date,
                "evidence": str(it.get("evidence", "")).strip()[:300]
            })
            
        return {
            "mode": "Meeting",
            "title": str(data.get("title", "Meeting Notes")).strip(),
            "headline": str(data.get("headline", "")).strip() or (data.get("summary", [""])[0] if data.get("summary") else ""),
            "summary": data.get("summary", [])[:3],
            "action_items": items,
            "attendees": data.get("attendees", []),
            "decisions": data.get("decisions", []),
            "discussion_points": data.get("discussion_points", []),
            "open_discussions": data.get("open_discussions", [])
        }
    except Exception as e:
        raise UserError(f"AI Analysis failed: {str(e)}")

def items_to_df(items):
    rows = [{"Done": False, "Task": it["task"], "Assignee": it["assignee"], "Priority": it["priority"], 
             "Status": "To Do", "Deadline": it["deadline_date"], "Implied Deadline": it["deadline_text"], 
             "Evidence": it["evidence"]} for it in items]
    df = pd.DataFrame(rows, columns=TASK_COLS)
    df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
    df["Done"] = df["Done"].astype(bool)
    return df

def run_pipeline(audio_bytes, filename, transcript_text, attendees, auto_purge, source):
    key = get_key()
    if not key:
        st.warning("Please add your Groq API key in the sidebar to continue.")
        return

    client = Groq(api_key=key)
    bar = st.progress(0, text="Starting...")
    try:
        transcript = transcript_text
        if audio_bytes is not None:
            bar.progress(25, text="Transcribing speech to text...")
            transcript = transcribe_audio(client, audio_bytes, filename)
            
        if not transcript.strip(): raise UserError("No speech detected.")
        
        bar.progress(65, text="Analysing with Llama 3.3...")
        minutes = extract_minutes(client, transcript, attendees)
        
        bar.progress(92, text="Building your task board...")
        st.session_state.transcript = transcript
        st.session_state.minutes = minutes
        st.session_state.source = source
        set_tasks(items_to_df(minutes["action_items"]))
        
        bar.progress(100, text="Done!")
        st.session_state.notice = "Analysis complete."
    except UserError as e:
        bar.empty(); st.error(str(e))
    finally:
        audio_bytes = None
    if auto_purge and source == "audio":
        st.session_state.upload_n += 1
        st.session_state.rec_n += 1
    st.rerun()

# ----------------------------------------------------------------------------
# UI Rendering Helpers
# ----------------------------------------------------------------------------
def clean_tasks(edited):
    df = edited.copy()
    df["Done"] = df["Done"].fillna(False).astype(bool)
    df["Task"] = df["Task"].fillna("").astype(str)
    df["Assignee"] = df["Assignee"].fillna("Unassigned").replace("", "Unassigned")
    df["Priority"] = df["Priority"].where(df["Priority"].isin(PRIORITIES), "Medium")
    df["Status"] = df["Status"].where(df["Status"].isin(STATUSES), "To Do")
    df.loc[df["Done"], "Status"] = "Done"
    df["Implied Deadline"] = df["Implied Deadline"].fillna("")
    df["Evidence"] = df["Evidence"].fillna("")
    df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
    return df.reset_index(drop=True)

def deadline_date(row):
    return pd.Timestamp(row["Deadline"]).date() if pd.notna(row["Deadline"]) else None

def due_state(row):
    if row["Done"]: return None
    d = deadline_date(row)
    if d is None: return ("No date", "d-nodate")
    delta = (d - date.today()).days
    if delta < 0: return ("Overdue", "d-overdue")
    if delta <= DUE_SOON_DAYS: return ("Due soon", "d-soon")
    return ("Upcoming", "d-upcoming")

def status_badge(row):
    if row["Done"] or row["Status"] == "Done": return ("Done", "s-done")
    return ("In Progress", "s-prog") if row["Status"] == "In Progress" else ("To Do", "s-todo")

def task_stats(df):
    valid = df[df["Task"].astype(str).str.strip() != ""]
    overdue = sum(1 for _, r in valid.iterrows() if (s := due_state(r)) and s[1] == "d-overdue")
    return {"total": len(valid), "done": int(valid["Done"].sum()), "prog": int(((~valid["Done"]) & (valid["Status"] == "In Progress")).sum()), "overdue": overdue}

def badges_html(row, with_status=True):
    out = f"<span class='badge b-{row['Priority']}'>{row['Priority']} priority</span>"
    if with_status:
        sl, sc = status_badge(row)
        out += f"<span class='badge {sc}'>{sl}</span>"
    ds = due_state(row)
    if ds: out += f"<span class='badge {ds[1]}'>{ds[0]}</span>"
    return out

def deadline_ui(row):
    d = deadline_date(row)
    parts = [f"📅 {d.strftime('%a, %d %b %Y')}" if d else "📅 No deadline"]
    if row["Implied Deadline"]: parts.append(f"<span class='ph'>said: “{html.escape(str(row['Implied Deadline']))}”</span>")
    return " &nbsp;·&nbsp; ".join(parts)

def task_card_html(row):
    cls = "task done" if row["Done"] else "task"
    ev = f"<div class='m ph'>💬 Evidence: ‘{html.escape(str(row['Evidence']))}’</div>" if row["Evidence"] else ""
    return f"<div class='{cls}'><div class='t'>{html.escape(str(row['Task']))}</div><div>{badges_html(row)}</div><div class='m'>👤 <b>{html.escape(str(row['Assignee']))}</b> &nbsp;·&nbsp; {deadline_ui(row)}</div>{ev}</div>"

def render_kanban(df):
    buckets = {"To Do": [], "In Progress": [], "Done": []}
    colors = {"To Do": "#64748b", "In Progress": "#2563eb", "Done": "#059669"}
    for _, r in df.iterrows():
        if not str(r["Task"]).strip(): continue
        key = "Done" if r["Done"] else (r["Status"] if r["Status"] in STATUSES else "To Do")
        buckets[key].append(r)
    cols = st.columns(3)
    for col, (name, rows) in zip(cols, buckets.items()):
        with col:
            st.markdown(f"<div class='kcol' style='border-color:{colors[name]}'>{name} &nbsp;({len(rows)})</div>", unsafe_allow_html=True)
            if not rows: st.caption("No tasks here.")
            for r in rows:
                st.markdown(f"<div class='kcard'><div class='t'>{html.escape(str(r['Task']))}</div><div>{badges_html(r, False)}</div><div class='m'>👤 {html.escape(str(r['Assignee']))}</div><div class='m'>{deadline_ui(r)}</div></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar & Header
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input("Groq API Key", type="password", key="key_GROQ_API_KEY", help="Powers both Transcription & Llama-3.3")
    st.caption("✅ Groq Key detected" if get_key() else "Waiting for API key...")
    st.divider()
    auto_purge = st.toggle("Clear uploaded audio after processing", value=True)
    st.button("🗑️ Clear meeting data", on_click=purge_all, use_container_width=True)

st.markdown("""<div class='mm-header'><div class='mm-logo'>🎙️</div><div><div class='mm-title'>MeetMind</div><div class='mm-sub'>AI Meeting Summarizer</div></div></div>
<div class='pipeline'><span class='step'>🎙️ Audio</span><span class='arrow'>→</span><span class='step'>⚡ Whisper</span><span class='arrow'>→</span><span class='step'>🧠 Llama 3.3</span><span class='arrow'>→</span><span class='step'>📋 Smart Minutes</span></div>""", unsafe_allow_html=True)

if st.session_state.notice:
    st.success(st.session_state.notice)
    st.session_state.notice = ""

with st.expander("🚀 Try the demo (no audio needed)", expanded=st.session_state.minutes is None):
    d1, d2, _ = st.columns([1.2, 1.2, 2])
    d1.button("📄 Load sample transcript", on_click=load_sample, use_container_width=True)
    if d2.button("▶️ Run full demo", type="primary", use_container_width=True):
        st.session_state.paste_text = SAMPLE_MEETING
        st.session_state.input_mode = "paste"
        run_pipeline(None, None, SAMPLE_MEETING, "", auto_purge, "text")

# ----------------------------------------------------------------------------
# Input Area
# ----------------------------------------------------------------------------
attendees = st.text_input("Attendee names (optional)", placeholder="Riya, Arjun, Meera")
MODES = [("upload", "📁 Upload Audio"), ("record", "🎤 Record Live"), ("paste", "📋 Paste Transcript")]
mode_cols = st.columns(3)
for col, (key, label) in zip(mode_cols, MODES):
    col.button(label, key=f"mode_{key}", on_click=lambda k=key: st.session_state.update({"input_mode": k}), use_container_width=True, type="primary" if st.session_state.input_mode == key else "secondary")

with st.container(border=True):
    mode = st.session_state.input_mode
    if mode == "upload":
        up = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"], key=f"up_{st.session_state.upload_n}")
        if up and st.button("✨ Generate Minutes", type="primary"):
            if up.size > MAX_MB * 1024 * 1024: st.error("File exceeds 25MB.")
            else: run_pipeline(up.getvalue(), up.name, None, attendees, auto_purge, "audio")
    elif mode == "record":
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record Live", key=f"rec_{st.session_state.rec_n}")
            if rec and st.button("✨ Generate Minutes", type="primary"):
                run_pipeline(rec.getvalue(), "rec.wav", None, attendees, auto_purge, "audio")
        else: st.info("Requires Streamlit 1.39+")
    else:
        c1, c2 = st.columns([5, 1.2])
        c2.button("Load sample", on_click=load_sample, use_container_width=True)
        txt = st.text_area("Paste raw transcript", key="paste_text", height=220)
        if st.button("✨ Analyze Transcript", type="primary"):
            if txt.strip(): run_pipeline(None, None, txt, attendees, auto_purge, "text")
            else: st.warning("Paste a transcript first.")

# ----------------------------------------------------------------------------
# Output Dashboard
# ----------------------------------------------------------------------------
m = st.session_state.minutes
st.markdown("---")
if not m:
    st.markdown("<div class='mm-card'><div class='mm-section'>No results yet</div>Awaiting input.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='mm-card mm-headline'><div class='lbl'>Key takeaway</div><div class='txt'>{html.escape(m['headline'])}</div><div class='ttl'>{html.escape(m['title'])}</div></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Attendees", len(m["attendees"]))
    k2.metric("Decisions", len(m["decisions"]))
    k3.metric("Action items", len(m["action_items"]))
    k4.metric("Open discussions", len(m["open_discussions"]))

    t_min, t_act, t_board, t_exp = st.tabs(["📝 Smart Minutes", "✅ Action Items", "🗂️ Task Board", "📤 Export"])

    with t_min:
        a, b = st.columns(2)
        with a: 
            st.markdown("<div class='mm-section'>Executive summary</div>", unsafe_allow_html=True)
            for x in m["summary"]: st.markdown(f"- {x}")
            st.markdown("<div class='mm-section'>Decisions made</div>", unsafe_allow_html=True)
            for x in (m["decisions"] or ["None"]): st.markdown(f"- {x}")
        with b:
            st.markdown("<div class='mm-section'>Discussion points</div>", unsafe_allow_html=True)
            for x in (m["discussion_points"] or ["None"]): st.markdown(f"- {x}")
            st.markdown("<div class='mm-section'>Open issues</div>", unsafe_allow_html=True)
            for x in (m["open_discussions"] or ["None"]): st.markdown(f"- {x}")

    with t_act:
        c_box, a_box = st.columns([1, 2])
        if len(st.session_state.tasks_base) == 0: st.button("➕ Add task", on_click=add_blank_task)
        edited = st.data_editor(st.session_state.tasks_base, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"te_{st.session_state.tasks_ver}",
            column_config={"Done": st.column_config.CheckboxColumn("Done", width="small"), "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITIES), "Status": st.column_config.SelectboxColumn("Status", options=STATUSES), "Deadline": st.column_config.DateColumn("Deadline", format="DD MMM YYYY")})
        st.session_state.tasks = clean_tasks(edited)
        tdf = st.session_state.tasks
        valid = tdf[tdf["Task"].astype(str).str.strip() != ""].copy()
        if not valid.empty:
            valid["_p"] = valid["Priority"].map(PRIORITY_RANK)
            valid = valid.sort_values(["Done", "_p"])
            st.markdown("".join(task_card_html(r) for _, r in valid.iterrows()), unsafe_allow_html=True)

    with t_board:
        render_kanban(st.session_state.tasks)

    with t_exp:
        md_text = f"# {m['title']}\n\n## Executive Summary\n" + "\n".join(f"- {x}" for x in m["summary"]) + "\n\n## Tasks\n"
        for _, r in valid.iterrows(): md_text += f"- [{'x' if r['Done'] else ' '}] {r['Task']} (Owner: {r['Assignee']})\n"
        st.download_button("⬇️ Download Markdown", md_text, "minutes.md", "text/markdown")
