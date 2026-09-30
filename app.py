"""
MeetMind - AI Meeting & Lecture Summarizer with Action-Item Tracker
BBIT Hackathon 2026 | Problem AI-03 (Groq-Only Edition)
"""
import html
import json
import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Config & State
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

def empty_tasks_df():
    df = pd.DataFrame(columns=TASK_COLS)
    df["Done"] = df["Done"].astype(bool)
    df["Deadline"] = pd.to_datetime(df["Deadline"])
    return df

defaults = {"transcript": "", "minutes": None, "source": "text", "tasks": empty_tasks_df(), 
            "tasks_base": empty_tasks_df(), "tasks_ver": 0, "upload_n": 0, "rec_n": 0, 
            "notice": "", "paste_text": "", "input_mode": "upload"}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

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

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
:root {--mm-primary:#4f46e5; --mm-border:rgba(128,128,128,.28); --mm-soft:rgba(128,128,128,.07);}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px;}
#MainMenu, footer {visibility: hidden;}
.mm-header {display:flex; align-items:center; gap:14px; margin-bottom:.3rem;}
.mm-logo {font-size:2.2rem; line-height:1;}
.mm-title {font-size:2rem; font-weight:800; letter-spacing:-.6px; color:var(--mm-primary); line-height:1.1;}
.mm-sub {font-size:1.02rem; font-weight:600; opacity:.85;}
.pipeline {display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:.9rem 0 1.1rem 0;}
.pipeline .step {border:1px solid var(--mm-border); border-radius:999px; padding:5px 14px; font-size:.85rem; font-weight:600; background:var(--mm-soft);}
.pipeline .arrow {opacity:.45; font-weight:700;}
.mm-card {border:1px solid var(--mm-border); border-radius:14px; padding:1rem 1.25rem; background:var(--mm-soft); box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:.8rem;}
.mm-headline {border-left:5px solid var(--mm-primary);}
.mm-headline .lbl {font-size:.75rem; font-weight:700; letter-spacing:.8px; opacity:.6; text-transform:uppercase;}
.mm-headline .txt {font-size:1.2rem; font-weight:650; margin-top:.2rem; line-height:1.4;}
.mm-headline .ttl {font-size:.92rem; opacity:.7; margin-top:.4rem;}
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
div[data-testid="stMetric"] {border:1px solid var(--mm-border); border-radius:14px; padding:.7rem 1rem; background:var(--mm-soft); box-shadow:0 1px 3px rgba(0,0,0,.05);}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# AI Core Pipeline
# ----------------------------------------------------------------------------
def transcribe_audio(client, audio_bytes, filename):
    res = client.audio.transcriptions.create(file=(filename, audio_bytes), model=GROQ_MODEL_AUDIO)
    return res.text.strip()

def extract_minutes(client, transcript, attendees):
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
    att = f"\nKnown attendees: {attendees}" if attendees else ""
    prompt = f"""You are an expert minute-taker. Return ONLY valid JSON matching this schema:
{schema}
Rules: Use ONLY information present in the transcript. Today's date is {today.isoformat()}. Priority: High/Medium/Low.{att}
Transcript:\n{transcript}"""
    
    res = client.chat.completions.create(
        model=GROQ_MODEL_TEXT, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.1
    )
    content = res.choices[0].message.content.strip()
    if content.startswith("```json"): content = content[7:-3]
    data = json.loads(content)
    
    items = []
    for it in data.get("action_items", []):
        if not str(it.get("task", "")).strip(): continue
        pr = str(it.get("priority", "Medium")).strip().title()
        dl_date = it.get("deadline_date")
        try: dl_date = datetime.strptime(str(dl_date).strip()[:10], "%Y-%m-%d").date().isoformat()
        except: dl_date = None
        
        items.append({
            "task": str(it["task"]).strip(), "assignee": str(it.get("assignee", "Unassigned")).strip() or "Unassigned",
            "priority": pr if pr in PRIORITIES else "Medium", "deadline_text": str(it.get("deadline_text", "")).strip(),
            "deadline_date": dl_date, "evidence": str(it.get("evidence", "")).strip()[:300]
        })
        
    return {
        "mode": "Meeting", "title": str(data.get("title", "Meeting Notes")).strip(),
        "headline": str(data.get("headline", "")).strip() or (data.get("summary", [""])[0] if data.get("summary") else ""),
        "summary": data.get("summary", [])[:3], "action_items": items,
        "attendees": data.get("attendees", []), "decisions": data.get("decisions", []),
        "discussion_points": data.get("discussion_points", []), "open_discussions": data.get("open_discussions", [])
    }

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
        st.error("❌ Missing Groq API Key! Please enter it in the sidebar settings on the left.")
        return

    progress_container = st.empty()
    progress_container.progress(5, text="Initializing...")
    
    try:
        client = Groq(api_key=key)
        transcript = transcript_text
        
        if audio_bytes is not None:
            progress_container.progress(25, text="Transcribing speech to text...")
            transcript = transcribe_audio(client, audio_bytes, filename)
            
        if not transcript or not transcript.strip(): 
            raise Exception("No speech detected. Please check your microphone and speak clearly.")
            
        progress_container.progress(65, text="Analysing with Llama 3.3...")
        # Brief pause to respect Groq rate limits
        time.sleep(1)
        minutes = extract_minutes(client, transcript, attendees)
        
        progress_container.progress(92, text="Building your task board...")
        st.session_state.transcript = transcript
        st.session_state.minutes = minutes
        st.session_state.source = source
        set_tasks(items_to_df(minutes["action_items"]))
        
        progress_container.progress(100, text="Done!")
        st.session_state.notice = "Analysis complete."
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Processing Error: {str(e)}")
        return # Critical Fix: This stops the app from wiping the error message off the screen

    # Only purge memory if the run was successful
    if auto_purge and source == "audio":
        st.session_state.upload_n += 1
        st.session_state.rec_n += 1
    st.rerun()

# ----------------------------------------------------------------------------
# UI Dashboard Rendering
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

def deadline_date(row): return pd.Timestamp(row["Deadline"]).date() if pd.notna(row["Deadline"]) else None

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
# Sidebar & Inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input("Groq API Key", type="password", key="key_GROQ_API_KEY", help="Powers both Transcription & Llama-3.3")
    if get_key(): st.success("✅ API Key loaded") 
    else: st.warning("⚠️ Waiting for API key...")
    st.divider()
    auto_purge = st.toggle("Clear uploaded audio after processing", value=True)
    st.button("🗑️ Clear meeting data", on_click=purge_all, use_container_width=True)

st.markdown("""<div class='mm-header'><div class='mm-logo'>🎙️</div><div><div class='mm-title'>MeetMind</div><div class='mm-sub'>AI Meeting Summarizer</div></div></div>
<div class='pipeline'><span class='step'>🎙️ Audio</span><span class='arrow'>→</span><span class='step'>⚡ Whisper</span><span class='arrow'>→</span><span class='step'>🧠 Llama 3.3</span><span class='arrow'>→</span><span class='step'>📋 Smart Minutes</span></div>""", unsafe_allow_html=True)

if st.session_state.notice:
    st.success(st.session_state.notice)
    st.session_state.notice = ""

attendees = st.text_input("Attendee names (optional)", placeholder="Riya, Arjun, Meera")
MODES = [("record", "🎤 Record Live"), ("upload", "📁 Upload Audio"), ("paste", "📋 Paste Transcript")]
mode_cols = st.columns(3)
for col, (key, label) in zip(mode_cols, MODES):
    col.button(label, key=f"mode_{key}", on_click=lambda k=key: st.session_state.update({"input_mode": k}), use_container_width=True, type="primary" if st.session_state.input_mode == key else "secondary")

with st.container(border=True):
    mode = st.session_state.input_mode
    
    # CRITICAL FIX: The Generate button is no longer nested inside the `if widget` condition
    if mode == "record":
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record Live Audio", key=f"rec_{st.session_state.rec_n}")
            if st.button("✨ Generate Minutes", type="primary", key="go_rec"):
                if rec is None: st.warning("Please record some audio first by clicking the microphone above.")
                else: run_pipeline(rec.getvalue(), "rec.wav", None, attendees, auto_purge, "audio")
        else: st.info("Requires Streamlit 1.39+")
        
    elif mode == "upload":
        up = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"], key=f"up_{st.session_state.upload_n}")
        if st.button("✨ Generate Minutes", type="primary", key="go_up"):
            if up is None: st.warning("Please upload a file first.")
            elif up.size > MAX_MB * 1024 * 1024: st.error("File exceeds 25MB limit.")
            else: run_pipeline(up.getvalue(), up.name, None, attendees, auto_purge, "audio")
            
    else:
        txt = st.text_area("Paste raw transcript", key="paste_text", height=220)
        if st.button("✨ Analyze Transcript", type="primary", key="go_txt"):
            if txt.strip(): run_pipeline(None, None, txt, attendees, auto_purge, "text")
            else: st.warning("Paste a transcript first.")

# ----------------------------------------------------------------------------
# Output Dashboard
# ----------------------------------------------------------------------------
m = st.session_state.minutes
st.markdown("---")
if not m:
    st.markdown("<div class='mm-card'><div class='mm-section'>No results yet</div>Awaiting input. Select a mode above and click Generate.</div>", unsafe_allow_html=True)
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
