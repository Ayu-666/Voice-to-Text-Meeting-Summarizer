"""
MeetMind - AI Meeting & Lecture Summarizer with Action-Item Tracker
BBIT Hackathon 2026 | Problem AI-03

Workflow : CAPTURE -> TRANSCRIBE -> SUMMARIZE -> EXTRACT ACTIONS -> TRACK TASKS -> EXPORT
Stack    : Streamlit + Groq Whisper (speech-to-text) + Gemini (structured extraction)
Run      : streamlit run app.py

Secrets (.streamlit/secrets.toml or Streamlit Cloud > Settings > Secrets):
    GROQ_API_KEY   = "gsk_..."
    GEMINI_API_KEY = "AIza..."
    GEMINI_MODEL   = "gemini-3.8-flash"   # optional: force one model (otherwise MeetMind auto-picks one your key can use)

Dependencies: streamlit>=1.40, requests, pandas, fpdf2   (no new dependencies vs. v1)
"""

import html
import json
import os
import re
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="MeetMind | AI Meeting & Lecture Summarizer",
    page_icon="🎙️",
    layout="wide",
)

MAX_MB = 25
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
# Tried in order; the first model your key can actually use is remembered for the session.
# (gemini-2.5-flash now returns 404 for many newer API keys, so it is only the last resort.)
GEMINI_FALLBACKS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
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
Riya: Done, Streamlit is final. Second, we'll use Groq Whisper for transcription and Gemini for the summaries.
Sam: I'm fine with that, both have free tiers.
Riya: Third decision, we freeze new features after Sunday and only fix bugs from then on.
Riya: Now action items. Arjun, please finish the database integration by this Friday. It's our top priority, the demo depends on it.
Arjun: Got it, I'll have it done by Friday.
Meera: I'll finish the export screens and share the final mockups with the team by next Monday.
Sam: I'll write the README and the docs folder and submit everything by October 10. Medium priority, but it's a hard deadline.
Riya: Sam, could you also book the seminar hall for a dry run? No rush, just sometime this month.
Sam: Sure, I'll do that.
Arjun: One more thing. Should we add PDF export? I'm not sure we have the time.
Riya: Let's keep that open and discuss it tomorrow. We also haven't decided who will present the demo to the judges.
Meera: Let's pick the presenter at the next meeting.
Riya: Okay, that's everything. Thanks, all.
"""

SAMPLE_LECTURE = """Professor: Today we're covering binary search trees. A binary search tree, or BST, is a binary tree where every key in a node's left subtree is smaller than the node, and every key in the right subtree is larger.
Professor: Searching works by comparing the target with the current node and going left or right. The cost of a search is proportional to the height of the tree. The height is the length of the longest path from the root down to a leaf.
Student: So is search always fast?
Professor: Good question. No. If you insert the keys 1, 2, 3, 4, 5 in that order, the tree becomes a skewed chain, and search degrades to linear time.
Professor: If you insert 50, 30, 70, 20, 40, the tree stays nicely balanced, and search is logarithmic. That's the contrast to remember.
Professor: To prevent the skewed case, we use self-balancing trees such as AVL trees. An AVL tree keeps the height difference between the two subtrees of any node at most one, and it uses rotations to restore this.
Student: Will rotations be on the exam?
Professor: Yes, definitely revise single and double rotations before the next class.
Professor: For homework, implement insertion and search for a BST in any language and submit it by next Thursday.
Professor: To summarize: a BST gives fast search only when it stays balanced, and AVL trees guarantee that. See you next week.
"""

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {--mm-primary:#4f46e5; --mm-border:rgba(128,128,128,.28); --mm-soft:rgba(128,128,128,.07);}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px;}
#MainMenu, footer {visibility: hidden;}

/* Header */
.mm-header {display:flex; align-items:center; gap:14px; margin-bottom:.3rem;}
.mm-logo {font-size:2.2rem; line-height:1;}
.mm-title {font-size:2rem; font-weight:800; letter-spacing:-.6px; color:var(--mm-primary); line-height:1.1;}
.mm-sub {font-size:1.02rem; font-weight:600; opacity:.85;}
.mm-tag {font-size:.92rem; opacity:.6; font-style:italic;}

/* Pipeline */
.pipeline {display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:.9rem 0 1.1rem 0;}
.pipeline .step {
    border:1px solid var(--mm-border); border-radius:999px; padding:5px 14px;
    font-size:.85rem; font-weight:600; background:var(--mm-soft);
}
.pipeline .arrow {opacity:.45; font-weight:700;}

/* Cards */
.mm-card {
    border:1px solid var(--mm-border); border-radius:14px; padding:1rem 1.25rem;
    background:var(--mm-soft); box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:.8rem;
}
.mm-headline {border-left:5px solid var(--mm-primary);}
.mm-headline .lbl {font-size:.75rem; font-weight:700; letter-spacing:.8px; opacity:.6; text-transform:uppercase;}
.mm-headline .txt {font-size:1.2rem; font-weight:650; margin-top:.2rem; line-height:1.4;}
.mm-headline .ttl {font-size:.92rem; opacity:.7; margin-top:.4rem;}
.mm-note {font-size:.82rem; opacity:.65; margin:.1rem 0 1rem 0;}
.mm-chips {display:flex; flex-wrap:wrap; gap:8px; margin:.4rem 0 .6rem 0;}
.mm-chip {border:1px solid var(--mm-border); border-radius:8px; padding:3px 10px; font-size:.78rem; opacity:.85;}
.mm-section {font-size:1.05rem; font-weight:700; margin:.2rem 0 .5rem 0;}

/* Badges */
.badge {
    display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
    font-weight:700; color:#fff; margin-right:6px; white-space:nowrap;
}
.b-High{background:#dc2626;} .b-Medium{background:#d97706;} .b-Low{background:#059669;}
.s-todo{background:#64748b;} .s-prog{background:#2563eb;} .s-done{background:#059669;}
.d-overdue{background:#dc2626;} .d-soon{background:#d97706;} .d-upcoming{background:#4f46e5;} .d-nodate{background:#94a3b8;}

/* Tasks */
.task {
    border:1px solid var(--mm-border); border-left:5px solid var(--mm-primary); border-radius:12px;
    padding:.75rem 1rem; margin-bottom:.6rem; background:var(--mm-soft);
}
.task.done {opacity:.6;} .task.done .t {text-decoration:line-through;}
.task .t {font-weight:650; font-size:1rem; margin-bottom:.4rem;}
.task .m {font-size:.84rem; opacity:.85; margin-top:.3rem;}
.task .ph {font-style:italic; opacity:.75;}
.kcol {font-weight:700; padding:.35rem .1rem; border-bottom:3px solid; margin-bottom:.7rem;}
.kcard {
    border:1px solid var(--mm-border); border-radius:12px; padding:.7rem .85rem;
    margin-bottom:.6rem; background:var(--mm-soft); box-shadow:0 1px 2px rgba(0,0,0,.05);
}
.kcard .t {font-weight:600; margin-bottom:.4rem;}
.kcard .m {font-size:.8rem; opacity:.85; margin-top:.3rem;}

/* Transcript */
.tbox {
    max-height:440px; overflow-y:auto; border:1px solid var(--mm-border); border-radius:12px;
    padding:1rem 1.3rem; background:var(--mm-soft); line-height:1.75; font-size:.97rem;
}
.tbox p {margin:0 0 .9rem 0;}
.tbox .spk {font-weight:700; margin-right:6px;}

/* Export */
.exp {border:1px solid var(--mm-border); border-radius:14px; padding:1rem 1.1rem; background:var(--mm-soft); margin-bottom:.6rem; min-height:118px;}
.exp .h {font-weight:700; font-size:1.02rem;}
.exp .d {font-size:.86rem; opacity:.75; margin-top:.25rem;}

/* Footer */
.mm-footer {text-align:center; font-size:.8rem; opacity:.5; margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--mm-border);}

div[data-testid="stMetric"] {
    border:1px solid var(--mm-border); border-radius:14px; padding:.7rem 1rem;
    background:var(--mm-soft); box-shadow:0 1px 3px rgba(0,0,0,.05);
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------


def empty_tasks_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=TASK_COLS)
    df["Done"] = df["Done"].astype(bool)
    df["Deadline"] = pd.to_datetime(df["Deadline"])
    return df


defaults = {
    "transcript": "",
    "minutes": None,
    "source": "text",  # "audio" or "text" - used for the transparency note
    "tasks": empty_tasks_df(),  # current (edited) tasks - used by Kanban + exports
    "tasks_base": empty_tasks_df(),  # data handed to the editor; only replaced programmatically
    "tasks_ver": 0,  # bumps the editor key whenever tasks_base is replaced
    "upload_n": 0,
    "rec_n": 0,
    "notice": "",
    "paste_text": "",
    "input_mode": "upload",
    "kind": "Meeting",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


class UserFacingError(Exception):
    """An error whose message is safe and helpful to show to the user."""


def secret(name: str) -> str:
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val or os.getenv(name) or st.session_state.get(f"key_{name}", "")


def set_tasks(df: pd.DataFrame):
    """Programmatically replace the task table (new results, manual add, purge)."""
    st.session_state.tasks = df
    st.session_state.tasks_base = df.copy()
    st.session_state.tasks_ver += 1


def purge_all():
    """Clear the transcript, generated results, tasks and pasted meeting text from the current
    session, and reset the audio widgets. API keys entered in the sidebar are NOT affected."""
    st.session_state.transcript = ""
    st.session_state.minutes = None
    st.session_state.paste_text = ""
    st.session_state.upload_n += 1
    st.session_state.rec_n += 1
    set_tasks(empty_tasks_df())


def load_sample():
    st.session_state.paste_text = SAMPLE_LECTURE if st.session_state.kind == "Lecture" else SAMPLE_MEETING
    st.session_state.input_mode = "paste"


def set_mode(mode: str):
    st.session_state.input_mode = mode


def add_blank_task():
    new = pd.DataFrame(
        [{"Done": False, "Task": "New task", "Assignee": "Unassigned", "Priority": "Medium",
          "Status": "To Do", "Deadline": pd.NaT, "Implied Deadline": "", "Evidence": ""}],
        columns=TASK_COLS,
    )
    cur = st.session_state.tasks.copy()
    cur.loc[cur["Status"] == "Done", "Status"] = "To Do"  # the editor only offers To Do / In Progress; Done is driven by the checkbox
    merged = new if cur.empty else pd.concat([cur, new], ignore_index=True)
    merged["Deadline"] = pd.to_datetime(merged["Deadline"], errors="coerce")
    merged["Done"] = merged["Done"].astype(bool)
    set_tasks(merged)


# ----------------------------------------------------------------------------
# AI calls (Groq Whisper + Gemini)
# ----------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes, filename: str, api_key: str) -> str:
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, audio_bytes)},
            data={"model": GROQ_MODEL, "response_format": "json", "temperature": 0},
            timeout=300,
        )
    except requests.RequestException:
        raise UserFacingError("Couldn't reach the Groq transcription service. Check your internet connection and try again.")
    if r.status_code == 401:
        raise UserFacingError("Groq rejected your API key. Double-check it in the sidebar (it should start with `gsk_`).")
    if r.status_code == 413:
        raise UserFacingError(f"Groq says this audio file is too large. Keep it under {MAX_MB} MB.")
    if r.status_code == 429:
        raise UserFacingError("Groq's free-tier rate limit was hit. Wait a minute and try again.")
    if r.status_code == 400:
        raise UserFacingError("Groq couldn't read this audio file. Try a different MP3/WAV export.")
    if r.status_code != 200:
        raise UserFacingError(f"Transcription failed (Groq status {r.status_code}). Please try again in a moment.")
    return r.json().get("text", "").strip()


def _parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def build_prompt(transcript: str, kind: str, attendees: str) -> str:
    today = date.today()
    shared_rules = f"""
GROUND RULES
- Use ONLY information that is actually present in the transcript. Never invent facts, names, tasks or deadlines.
- If a section has nothing to report, return an empty list for it.
- Today's date is {today.isoformat()} ({today.strftime('%A')}).

DEADLINE RULES (important)
- "deadline_text": copy the exact words from the transcript that mention timing (e.g. "this Friday",
  "next Monday", "by October 10"). Use "" if the transcript gives no timing for that task.
- "deadline_date": YYYY-MM-DD, but ONLY when the wording clearly points to one specific date.
  "this <weekday>" = the next occurrence of that weekday on or after today.
  "next <weekday>" = the first occurrence of that weekday after today.
  "by <month> <day>" = that date in the current year (next year if it already passed).
  For vague wording ("soon", "sometime this month", "ASAP", "later") or no timing at all, use null.
  Never guess. When unsure, use null and keep the original phrase in "deadline_text".
- Assign a task to a person only if the transcript names who does it; otherwise use "Unassigned".
- Priority: High / Medium / Low, based only on urgency cues in the transcript (default Medium).
- "deadline_text" must appear WORD-FOR-WORD in the transcript, otherwise use "" and null.

EVIDENCE RULES
- Every action item needs "evidence": the exact short sentence or phrase copied word-for-word from the
  transcript that shows this task was assigned or agreed. Do not paraphrase, merge or reword it.
- If you cannot point to such words, use "" for evidence. Never invent evidence.
"""
    if kind == "Lecture":
        schema = """{
  "title": "short descriptive title of the lecture",
  "headline": "ONE sentence: the single most important idea of the lecture",
  "summary": ["exactly 3 concise bullets summarising the lecture"],
  "main_concepts": [{"concept": "name", "explanation": "1-2 sentence explanation from the lecture"}],
  "definitions": [{"term": "term", "definition": "definition as given in the lecture"}],
  "key_takeaways": ["what a student should remember"],
  "examples": ["important examples the lecturer used"],
  "revision_topics": ["topics or questions the student should revise, especially ones the lecturer flagged"],
  "action_items": [{"task": "...", "assignee": "...", "priority": "High|Medium|Low", "deadline_text": "...", "deadline_date": "YYYY-MM-DD or null", "evidence": "exact words from the transcript"}]
}
For "action_items" in a lecture include ONLY explicit assignments, homework or submissions the lecturer mentioned."""
        role = "You are an expert note-taker for university lectures."
    else:
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
        role = "You are an expert minute-taker for meetings."

    att = f"\nKnown attendees (optional hint): {attendees}" if attendees and kind != "Lecture" else ""
    return f"""{role} Analyse the transcript and return ONLY valid JSON matching this schema:
{schema}
{shared_rules}{att}

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\""""


class ModelUnavailable(UserFacingError):
    """This model can't be used with the current key right now (not found / quota) - try the next one."""


def _api_detail(r) -> str:
    try:
        return str(r.json().get("error", {}).get("message", ""))[:200]
    except (ValueError, AttributeError):
        return r.text[:200]


def call_gemini(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        r = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
            },
            timeout=180,
        )
    except requests.RequestException:
        raise UserFacingError("Couldn't reach Gemini. Check your internet connection and try again.")
    if r.status_code in (401, 403) or (r.status_code == 400 and "API_KEY" in r.text.upper()):
        raise UserFacingError("Gemini rejected your API key. Double-check it in the sidebar (free keys: aistudio.google.com).")
    if r.status_code == 404:
        raise ModelUnavailable(f"`{model}` isn't available for your API key.")
    if r.status_code == 429:
        raise ModelUnavailable(f"The free-tier rate limit or quota for `{model}` was hit.")
    if r.status_code >= 500:
        raise UserFacingError("Gemini is temporarily unavailable. Please try again shortly.")
    if r.status_code != 200:
        detail = _api_detail(r)
        raise UserFacingError(f"Gemini returned an error (status {r.status_code})" + (f": {detail}" if detail else "."))
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError):
        raise UserFacingError("Gemini returned an empty answer (it may have been filtered). Please click Analyze again.")


def call_gemini_any(prompt: str, api_key: str, preferred: str = "") -> str:
    """Call Gemini, automatically moving to the next model if one is unavailable for this key."""
    order = []
    for mdl in [preferred, st.session_state.get("gemini_ok"), *GEMINI_FALLBACKS]:
        if mdl and mdl not in order:
            order.append(mdl)
    tried, last = [], None
    for mdl in order:
        try:
            text = call_gemini(prompt, api_key, mdl)
            st.session_state.gemini_ok = mdl
            return text
        except ModelUnavailable as e:
            tried.append(mdl)
            last = e
            if st.session_state.get("gemini_ok") == mdl:
                st.session_state.gemini_ok = None
    raise UserFacingError(
        f"{last} Models tried: {', '.join(tried)}. Wait a minute and try again, or set `GEMINI_MODEL` to a model "
        "your key can use (see ai.google.dev/gemini-api/docs/models)."
    )


def _strs(val) -> list:
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()]


def _pairs(val, k1: str, k2: str) -> list:
    out = []
    if not isinstance(val, list):
        return out
    for it in val:
        if isinstance(it, dict):
            a, b = str(it.get(k1, "")).strip(), str(it.get(k2, "")).strip()
        else:
            a, b = str(it).strip(), ""
        if a:
            out.append({k1: a, k2: b})
    return out


VAGUE_RE = re.compile(
    r"\b(sometime|some time|soon|asap|later|eventually|whenever|someday|at some point|no rush|"
    r"this month|next month|this week|next week|few days|shortly|when (?:we|you|i) can)\b",
    re.IGNORECASE,
)


def _norm_text(s) -> str:
    s = str(s).lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def in_transcript(snippet, transcript_norm: str) -> bool:
    """True if the snippet appears in the (already normalised) transcript, case-insensitively."""
    sn = _norm_text(snippet).strip(" .,;:!?\"'")
    return bool(sn) and sn in transcript_norm


def clean_date(value, phrase: str, transcript_norm: str):
    """Keep a date only if it is valid AND its original phrase really appears in the transcript."""
    if not value or not phrase or not in_transcript(phrase, transcript_norm):
        return None
    if VAGUE_RE.search(phrase):  # vague wording never earns a concrete date
        return None
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def normalize_minutes(data, kind: str, transcript: str) -> dict:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("unexpected JSON shape")
    tnorm = _norm_text(transcript)
    items = []
    for it in data.get("action_items") or []:
        if not isinstance(it, dict) or not str(it.get("task", "")).strip():
            continue
        phrase = str(it.get("deadline_text") or "").strip()
        if phrase and not in_transcript(phrase, tnorm):
            phrase = ""  # the phrase isn't in the transcript, so it can't be shown as "said"
        evidence = str(it.get("evidence") or "").strip()
        if not in_transcript(evidence, tnorm):
            evidence = ""  # never keep evidence that can't be found in the transcript
        evidence = evidence[:300]
        pr = str(it.get("priority", "Medium")).strip().title()
        items.append(
            {
                "task": str(it["task"]).strip(),
                "assignee": str(it.get("assignee") or "Unassigned").strip() or "Unassigned",
                "priority": pr if pr in PRIORITIES else "Medium",
                "deadline_text": phrase,
                "deadline_date": clean_date(it.get("deadline_date"), phrase, tnorm),
                "evidence": evidence,
            }
        )
    m = {
        "mode": kind,
        "title": str(data.get("title") or f"{kind} Notes").strip(),
        "headline": str(data.get("headline") or "").strip(),
        "summary": _strs(data.get("summary"))[:3],
        "action_items": items,
    }
    if kind == "Lecture":
        m["main_concepts"] = _pairs(data.get("main_concepts"), "concept", "explanation")
        m["definitions"] = _pairs(data.get("definitions"), "term", "definition")
        m["key_takeaways"] = _strs(data.get("key_takeaways"))
        m["examples"] = _strs(data.get("examples"))
        m["revision_topics"] = _strs(data.get("revision_topics"))
    else:
        m["attendees"] = _strs(data.get("attendees"))
        m["decisions"] = _strs(data.get("decisions"))
        m["discussion_points"] = _strs(data.get("discussion_points"))
        m["open_discussions"] = _strs(data.get("open_discussions"))
    if not m["headline"] and m["summary"]:
        m["headline"] = m["summary"][0]
    return m


def extract_minutes(transcript: str, kind: str, attendees: str, api_key: str, model: str) -> dict:
    prompt = build_prompt(transcript, kind, attendees)
    for _ in range(2):  # one automatic retry if the AI returns malformed JSON
        raw = call_gemini_any(prompt, api_key, model)
        try:
            return normalize_minutes(_parse_json(raw), kind, transcript)
        except (ValueError, TypeError):
            continue
    raise UserFacingError("The AI's answer couldn't be read (malformed response). Please click Analyze again - it usually works on retry.")


def items_to_df(items: list) -> pd.DataFrame:
    rows = [
        {
            "Done": False,
            "Task": it["task"],
            "Assignee": it["assignee"],
            "Priority": it["priority"],
            "Status": "To Do",
            "Deadline": it["deadline_date"],
            "Implied Deadline": it["deadline_text"],
            "Evidence": it["evidence"],
        }
        for it in items
    ]
    df = pd.DataFrame(rows, columns=TASK_COLS)
    df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
    df["Done"] = df["Done"].astype(bool)
    return df


# ----------------------------------------------------------------------------
# Pipeline with progress bar
# ----------------------------------------------------------------------------
def run_pipeline(audio_bytes, filename, transcript_text, kind, attendees, auto_purge, source):
    groq_key, gem_key = secret("GROQ_API_KEY"), secret("GEMINI_API_KEY")
    model = secret("GEMINI_MODEL")  # optional override; empty = auto-select
    had_audio = audio_bytes is not None  # remember before the raw audio reference is dropped

    missing = []
    if not gem_key:
        missing.append("a **Gemini API key** (free at aistudio.google.com) to generate the minutes")
    if audio_bytes is not None and not groq_key:
        missing.append("a **Groq API key** (free at console.groq.com) to transcribe audio")
    if missing:
        st.warning(
            "To continue, add " + " and ".join(missing) + ". Paste the key(s) in the sidebar on the left "
            "or add them to Streamlit secrets. Pasted transcripts only need the Gemini key."
        )
        return

    bar = st.progress(0, text="Starting...")
    try:
        transcript = transcript_text
        if audio_bytes is not None:
            bar.progress(10, text="Uploading audio to Whisper...")
            bar.progress(25, text="Transcribing speech to text...")
            transcript = transcribe_audio(audio_bytes, filename, groq_key)
            bar.progress(55, text="Transcription complete.")
        if not transcript or not transcript.strip():
            raise UserFacingError("No speech was detected. Check that the recording has audible speech and try again.")
        if len(transcript.split()) < 10:
            raise UserFacingError("That transcript is too short to summarise. Add a few more sentences (10+ words).")
        bar.progress(65, text=f"Analysing with Gemini ({kind.lower()} mode)...")
        minutes = extract_minutes(transcript, kind, attendees, gem_key, model)
        bar.progress(92, text="Building your task board...")
        st.session_state.transcript = transcript
        st.session_state.minutes = minutes
        st.session_state.source = source
        set_tasks(items_to_df(minutes["action_items"]))
        bar.progress(100, text="Done!")
        st.session_state.notice = f"{kind} analysis complete."
    except UserFacingError as e:
        bar.empty()
        st.error(str(e))
        return
    except Exception:  # noqa: BLE001
        bar.empty()
        st.error("Something unexpected happened while processing. Please try again. "
                 "If it keeps failing, try the sample transcript to check your API keys.")
        return
    finally:
        audio_bytes = None  # drop our reference to the raw audio immediately

    if auto_purge and had_audio:
        st.session_state.upload_n += 1
        st.session_state.rec_n += 1
        st.session_state.notice += " The uploaded audio was cleared from this session."
    st.rerun()


# ----------------------------------------------------------------------------
# Task helpers
# ----------------------------------------------------------------------------
def clean_tasks(edited: pd.DataFrame) -> pd.DataFrame:
    df = edited.copy()
    df["Done"] = df["Done"].fillna(False).astype(bool)
    df["Task"] = df["Task"].fillna("").astype(str)
    df["Assignee"] = df["Assignee"].fillna("Unassigned").replace("", "Unassigned")
    df["Priority"] = df["Priority"].where(df["Priority"].isin(PRIORITIES), "Medium")
    df["Status"] = df["Status"].where(df["Status"].isin(STATUSES), "To Do")
    df.loc[df["Done"], "Status"] = "Done"  # a ticked task is always Done, whatever the Status cell says
    df["Implied Deadline"] = df["Implied Deadline"].fillna("")
    df["Evidence"] = df["Evidence"].fillna("")
    df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
    return df.reset_index(drop=True)


def deadline_date(row):
    return pd.Timestamp(row["Deadline"]).date() if pd.notna(row["Deadline"]) else None


def due_state(row):
    """(label, css) for Overdue / Due Soon / Upcoming / No date; None if the task is done."""
    if row["Done"]:
        return None
    d = deadline_date(row)
    if d is None:
        return ("No date", "d-nodate")
    delta = (d - date.today()).days
    if delta < 0:
        return ("Overdue", "d-overdue")
    if delta <= DUE_SOON_DAYS:
        return ("Due soon", "d-soon")
    return ("Upcoming", "d-upcoming")


def status_badge(row):
    if row["Done"] or row["Status"] == "Done":
        return ("Done", "s-done")
    return ("In Progress", "s-prog") if row["Status"] == "In Progress" else ("To Do", "s-todo")


def task_stats(df: pd.DataFrame) -> dict:
    valid = df[df["Task"].astype(str).str.strip() != ""]
    overdue = sum(1 for _, r in valid.iterrows() if (s := due_state(r)) and s[1] == "d-overdue")
    return {
        "total": len(valid),
        "done": int(valid["Done"].sum()),
        "prog": int(((~valid["Done"]) & (valid["Status"] == "In Progress")).sum()),
        "overdue": overdue,
    }


def deadline_ui(row) -> str:
    d = deadline_date(row)
    parts = [f"📅 {d.strftime('%a, %d %b %Y')}" if d else "📅 No deadline"]
    if row["Implied Deadline"]:
        parts.append(f"<span class='ph'>said: “{html.escape(str(row['Implied Deadline']))}”</span>")
    return " &nbsp;·&nbsp; ".join(parts)


def badges_html(row, with_status=True) -> str:
    out = f"<span class='badge b-{row['Priority']}'>{row['Priority']} priority</span>"
    if with_status:
        sl, sc = status_badge(row)
        out += f"<span class='badge {sc}'>{sl}</span>"
    ds = due_state(row)
    if ds:
        out += f"<span class='badge {ds[1]}'>{ds[0]}</span>"
    return out


def task_card_html(row) -> str:
    cls = "task done" if row["Done"] else "task"
    ev = (f"<div class='m ph'>💬 Evidence: ‘{html.escape(str(row['Evidence']))}’</div>"
          if row["Evidence"] else "")
    return (
        f"<div class='{cls}'><div class='t'>{html.escape(str(row['Task']))}</div>"
        f"<div>{badges_html(row)}</div>"
        f"<div class='m'>👤 <b>{html.escape(str(row['Assignee']))}</b> &nbsp;·&nbsp; {deadline_ui(row)}</div>{ev}</div>"
    )


def render_kanban(df: pd.DataFrame):
    buckets = {"To Do": [], "In Progress": [], "Done": []}
    colors = {"To Do": "#64748b", "In Progress": "#2563eb", "Done": "#059669"}
    for _, r in df.iterrows():
        if not str(r["Task"]).strip():
            continue
        key = "Done" if r["Done"] else (r["Status"] if r["Status"] in STATUSES else "To Do")
        buckets[key].append(r)
    cols = st.columns(3)
    for col, (name, rows) in zip(cols, buckets.items()):
        with col:
            st.markdown(f"<div class='kcol' style='border-color:{colors[name]}'>{name} &nbsp;({len(rows)})</div>",
                        unsafe_allow_html=True)
            if not rows:
                st.caption("No tasks here.")
            for r in rows:
                st.markdown(
                    f"<div class='kcard'><div class='t'>{html.escape(str(r['Task']))}</div>"
                    f"<div>{badges_html(r, with_status=False)}</div>"
                    f"<div class='m'>👤 {html.escape(str(r['Assignee']))}</div>"
                    f"<div class='m'>{deadline_ui(r)}</div></div>",
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------------
# Transcript rendering
# ----------------------------------------------------------------------------
SPEAKER_RE = re.compile(r"^\s*([A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,2}):\s+(.+)$")
SPEAKER_COLORS = ["#4f46e5", "#0d9488", "#d97706", "#db2777", "#2563eb", "#7c3aed"]


def render_transcript_html(text: str):
    """Returns (html, detected_speakers). Speakers are shown only if the transcript itself has 'Name:' labels."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = Counter(m.group(1) for l in lines if (m := SPEAKER_RE.match(l)))
    speakers = [s for s, c in counts.items() if c >= 2]
    colors = {s: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, s in enumerate(speakers)}

    if len(lines) == 1 and not speakers:  # typical Whisper output: one long blob -> readable paragraphs
        sentences = re.split(r"(?<=[.!?])\s+", lines[0])
        lines = [" ".join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]

    blocks = []
    for l in lines:
        m = SPEAKER_RE.match(l)
        if m and m.group(1) in colors:
            name = html.escape(m.group(1))
            blocks.append(f"<p><span class='spk' style='color:{colors[m.group(1)]}'>{name}:</span>{html.escape(m.group(2))}</p>")
        else:
            blocks.append(f"<p>{html.escape(l)}</p>")
    return f"<div class='tbox'>{''.join(blocks)}</div>", speakers


# ----------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------
def export_deadline(row) -> str:
    d = deadline_date(row)
    parts = []
    if d:
        parts.append(d.strftime("%d %b %Y"))
    if row["Implied Deadline"]:
        parts.append(f'(said: "{row["Implied Deadline"]}")')
    return " ".join(parts) or "-"


def minutes_sections(m: dict):
    """Mode-specific sections as (heading, [plain-text items]) - shared by Markdown and PDF."""
    if m["mode"] == "Lecture":
        return [
            ("Lecture Summary", m["summary"]),
            ("Main Concepts", [f"{c['concept']}: {c['explanation']}" if c["explanation"] else c["concept"] for c in m["main_concepts"]]),
            ("Important Definitions", [f"{d['term']}: {d['definition']}" if d["definition"] else d["term"] for d in m["definitions"]]),
            ("Key Takeaways", m["key_takeaways"]),
            ("Important Examples", m["examples"]),
            ("Topics to Revise", m["revision_topics"]),
        ]
    return [
        ("Executive Summary", m["summary"]),
        ("Decisions Made", m["decisions"]),
        ("Discussion Points", m["discussion_points"]),
        ("Open Issues", m["open_discussions"]),
    ]


def build_markdown(m: dict, df: pd.DataFrame) -> str:
    L = [f"# {m['title']}", f"*{m['mode']} notes · generated {date.today():%d %b %Y} with speech-to-text + AI analysis*", ""]
    if m["headline"]:
        L += [f"> **Key takeaway:** {m['headline']}", ""]
    if m.get("attendees"):
        L += [f"**Attendees:** {', '.join(m['attendees'])}", ""]
    for head, items in minutes_sections(m):
        L += [f"## {head}"] + ([f"- {i}" for i in items] or ["- Nothing detected in the transcript."]) + [""]
    L.append("## Action Items")
    valid = df[df["Task"].astype(str).str.strip() != ""]
    if valid.empty:
        L.append("No explicit action items detected.")
    else:
        L += ["", "| Done | Task | Assignee | Priority | Status | Deadline | Evidence |", "|---|---|---|---|---|---|---|"]
        for _, r in valid.iterrows():
            c = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
            L.append(f"| {'[x]' if r['Done'] else '[ ]'} | {c(r['Task'])} | {c(r['Assignee'])} | {r['Priority']} "
                     f"| {status_badge(r)[0]} | {c(export_deadline(r))} | {c(r['Evidence']) or '-'} |")
    return "\n".join(L) + "\n"


def build_pdf(m: dict, df: pd.DataFrame):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    def l1(s):
        return str(s).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    def write(text, style="", size=11, h=6):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, h, l1(text), new_x="LMARGIN", new_y="NEXT")

    def head(text):
        pdf.ln(3)
        write(text, "B", 13, 8)

    write(m["title"], "B", 18, 10)
    write(f"{m['mode']} notes - generated {date.today():%d %b %Y} with speech-to-text + AI analysis", "I", 9)
    if m["headline"]:
        pdf.ln(2)
        write("Key takeaway: " + m["headline"], "B", 11)
    if m.get("attendees"):
        write("Attendees: " + ", ".join(m["attendees"]))
    for title, items in minutes_sections(m):
        head(title)
        for i in items or ["Nothing detected in the transcript."]:
            write(f"- {i}")
    head("Action Items")
    valid = df[df["Task"].astype(str).str.strip() != ""]
    if valid.empty:
        write("No explicit action items detected.")
    for n, (_, r) in enumerate(valid.iterrows(), 1):
        write(f"{n}. [{'x' if r['Done'] else ' '}] {r['Task']}", "B", 11)
        write(f"    Owner: {r['Assignee']}  |  Priority: {r['Priority']}  |  Status: {status_badge(r)[0]}  |  Due: {export_deadline(r)}", "", 10)
        if r["Evidence"]:
            write(f'    Evidence: "{r["Evidence"]}"', "I", 9)
    return bytes(pdf.output())


def build_ics(df: pd.DataFrame, title: str):
    def esc(s):
        return str(s).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    L = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MeetMind//Hackathon 2026//EN", "CALSCALE:GREGORIAN"]
    count = 0
    for _, r in df.iterrows():
        d = deadline_date(r)
        if d is None or not str(r["Task"]).strip():
            continue
        desc = f"From: {title} | Priority: {r['Priority']}"
        if r["Implied Deadline"]:
            desc += f' | Said in transcript: "{r["Implied Deadline"]}"'
        L += [
            "BEGIN:VEVENT",
            f"UID:{uuid.uuid4()}@meetmind",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{d:%Y%m%d}",
            f"DTEND;VALUE=DATE:{d + timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{esc('[' + str(r['Assignee']) + '] ' + str(r['Task']))}",
            f"DESCRIPTION:{esc(desc)}",
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            "DESCRIPTION:Task due tomorrow",
            "TRIGGER:-P1D",
            "END:VALARM",
            "END:VEVENT",
        ]
        count += 1
    L.append("END:VCALENDAR")
    return "\r\n".join(L) + "\r\n", count


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def bullet_card(title: str, items: list, empty: str):
    with st.container(border=True):
        st.markdown(f"<div class='mm-section'>{title}</div>", unsafe_allow_html=True)
        if items:
            for i in items:
                st.markdown(f"- {i}")
        else:
            st.caption(empty)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input("Groq API key (speech-to-text)", type="password", key="key_GROQ_API_KEY",
                  help="Free at console.groq.com. Only needed for audio input.")
    st.caption("✅ Groq key ready" if secret("GROQ_API_KEY") else "Needed for audio upload / recording.")
    st.text_input("Gemini API key (analysis)", type="password", key="key_GEMINI_API_KEY",
                  help="Free at aistudio.google.com. Needed for every analysis.")
    st.caption("✅ Gemini key ready" if secret("GEMINI_API_KEY") else "Needed for every analysis.")
    st.divider()
    st.markdown("### 🔒 Privacy")
    auto_purge = st.toggle("Clear uploaded audio after processing", value=True)
    st.caption(
        "MeetMind keeps audio in memory only and never saves it to disk. Audio is sent to Groq for "
        "transcription and the transcript text to Google Gemini for analysis, so their own data policies apply."
    )
    st.button("🗑️ Clear meeting data", on_click=purge_all, use_container_width=True,
              help="Clears the transcript, results, tasks and pasted text. Your API keys are kept.")

# ----------------------------------------------------------------------------
# Header + pipeline
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class='mm-header'>
  <div class='mm-logo'>🎙️</div>
  <div>
    <div class='mm-title'>MeetMind</div>
    <div class='mm-sub'>AI Meeting &amp; Lecture Summarizer</div>
  </div>
</div>
<div class='mm-tag'>From conversation to clear actions.</div>
<div class='pipeline'>
  <span class='step'>🎙️ Audio</span><span class='arrow'>→</span>
  <span class='step'>⚡ Whisper</span><span class='arrow'>→</span>
  <span class='step'>🧠 Gemini</span><span class='arrow'>→</span>
  <span class='step'>📋 Smart Minutes</span><span class='arrow'>→</span>
  <span class='step'>✅ Action Tracker</span>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.notice:
    st.success(st.session_state.notice)
    st.session_state.notice = ""

# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
with st.expander("🚀 Try the demo (no audio needed)", expanded=st.session_state.minutes is None):
    st.markdown(
        "**1.** Load the sample transcript  →  **2.** Click *Analyze Transcript*  →  "
        "**3.** Explore the minutes and action tracker. The sample has several speakers, decisions, "
        "priorities, deadlines and an open discussion. Only the Gemini key is needed."
    )
    d1, d2, _ = st.columns([1.2, 1.2, 2])
    d1.button("📄 Load sample transcript", on_click=load_sample, use_container_width=True)
    if d2.button("▶️ Run full demo", type="primary", use_container_width=True):
        sample = SAMPLE_LECTURE if st.session_state.kind == "Lecture" else SAMPLE_MEETING
        st.session_state.paste_text = sample
        st.session_state.input_mode = "paste"
        run_pipeline(None, None, sample, st.session_state.kind, "", auto_purge, "text")

# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------
st.markdown("## Capture your meeting")
st.radio("What are you summarizing?", ["Meeting", "Lecture"], horizontal=True, key="kind",
         help="Meeting mode extracts decisions, discussion points and open issues. "
              "Lecture mode extracts concepts, definitions, takeaways and revision topics.")
kind = st.session_state.kind
attendees = ""
if kind == "Meeting":
    attendees = st.text_input("Attendee names (optional)", placeholder="Riya, Arjun, Meera",
                              help="Helps the AI assign tasks to the right people.")

MODES = [
    ("upload", "📁 Upload Audio", "MP3 or WAV recording"),
    ("record", "🎤 Record Live", "Use your microphone"),
    ("paste", "📋 Paste Transcript", "Text you already have"),
]
mode_cols = st.columns(3)
for col, (key, label, _) in zip(mode_cols, MODES):
    col.button(label, key=f"mode_{key}", on_click=set_mode, args=(key,), use_container_width=True,
               type="primary" if st.session_state.input_mode == key else "secondary")

cta_audio = "Generate Meeting Minutes" if kind == "Meeting" else "Generate Lecture Notes"

with st.container(border=True):
    mode = st.session_state.input_mode
    if mode == "upload":
        st.markdown(f"<div class='mm-chips'><span class='mm-chip'>Supported: MP3, WAV</span>"
                    f"<span class='mm-chip'>Max size: {MAX_MB} MB</span>"
                    f"<span class='mm-chip'>🔒 Audio is not saved by MeetMind</span></div>", unsafe_allow_html=True)
        up = st.file_uploader("Drop your recording here", type=["mp3", "wav"], key=f"up_{st.session_state.upload_n}",
                              label_visibility="collapsed")
        if up is not None:
            if up.size > MAX_MB * 1024 * 1024:
                st.error(f"This file is {up.size / 1e6:.1f} MB, which is over the {MAX_MB} MB limit. "
                         "Try trimming it or exporting at a lower bitrate.")
            else:
                st.audio(up)
                if st.button(f"✨ {cta_audio}", key="go_up", type="primary"):
                    run_pipeline(up.getvalue(), up.name, None, kind, attendees, auto_purge, "audio")
        else:
            st.caption("Tip: a 30-60 second clip is perfect for a quick demo.")

    elif mode == "record":
        st.markdown("<div class='mm-chips'><span class='mm-chip'>Speak for 30+ seconds</span>"
                    f"<span class='mm-chip'>Max size: {MAX_MB} MB</span>"
                    "<span class='mm-chip'>🔒 Audio is not saved by MeetMind</span></div>", unsafe_allow_html=True)
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record from your microphone", key=f"rec_{st.session_state.rec_n}",
                                 label_visibility="collapsed")
            if rec is not None:
                if rec.size > MAX_MB * 1024 * 1024:
                    st.error(f"This recording is over the {MAX_MB} MB limit. Please record a shorter clip.")
                elif st.button(f"✨ {cta_audio}", key="go_rec", type="primary"):
                    run_pipeline(rec.getvalue(), "recording.wav", None, kind, attendees, auto_purge, "audio")
            else:
                st.caption("Allow microphone access in your browser, press record, speak, then stop.")
        else:
            st.info("Live recording needs Streamlit 1.39 or newer. Upgrade with `pip install -U streamlit`, "
                    "or use Upload Audio / Paste Transcript instead.")

    else:
        st.markdown("<div class='mm-chips'><span class='mm-chip'>Plain text</span>"
                    "<span class='mm-chip'>Tip: use “Name: text” lines to keep speakers visible</span></div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns([5, 1.2])
        c2.button("Load sample", on_click=load_sample, use_container_width=True)
        txt = st.text_area("Paste raw transcript", key="paste_text", height=220,
                           placeholder="Riya: Let's start...\nArjun: Quick update...",
                           label_visibility="collapsed")
        if st.button("✨ Analyze Transcript", key="go_txt", type="primary"):
            if not txt.strip():
                st.warning("Paste a transcript first, or click **Load sample** to try the demo.")
            else:
                run_pipeline(None, None, txt, kind, attendees, auto_purge, "text")

# ----------------------------------------------------------------------------
# Results dashboard
# ----------------------------------------------------------------------------
m = st.session_state.minutes
st.markdown("---")
if not m:
    st.markdown(
        "<div class='mm-card'><div class='mm-section'>No results yet</div>"
        "Choose an input above (upload, record, or paste a transcript) and click the generate button. "
        "Your summary, action items and exports will appear here. "
        "Not sure where to begin? Open <b>Try the demo</b> at the top.</div>",
        unsafe_allow_html=True,
    )
else:
    is_lecture = m["mode"] == "Lecture"
    st.markdown(
        f"<div class='mm-card mm-headline'><div class='lbl'>{'Main idea' if is_lecture else 'Key takeaway'}</div>"
        f"<div class='txt'>{html.escape(m['headline'] or 'No headline was generated.')}</div>"
        f"<div class='ttl'>{html.escape(m['title'])} &nbsp;·&nbsp; {m['mode']} mode</div></div>",
        unsafe_allow_html=True,
    )
    origin = "your recording using speech-to-text + AI analysis" if st.session_state.source == "audio" \
        else "your transcript using AI analysis"
    st.markdown(f"<div class='mm-note'>✨ Generated from {origin}. AI can make mistakes, so please verify "
                "important details against the transcript.</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    if is_lecture:
        k1.metric("Main concepts", len(m["main_concepts"]))
        k2.metric("Definitions", len(m["definitions"]))
        k3.metric("Key takeaways", len(m["key_takeaways"]))
        k4.metric("Topics to revise", len(m["revision_topics"]))
    else:
        k1.metric("Attendees", len(m["attendees"]))
        k2.metric("Decisions", len(m["decisions"]))
        k3.metric("Action items", len(m["action_items"]))
        k4.metric("Open discussions", len(m["open_discussions"]))

    t_min, t_act, t_board, t_tr, t_exp = st.tabs(
        ["📝 Smart " + ("Notes" if is_lecture else "Minutes"), "✅ Action Items", "🗂️ Task Board", "📜 Transcript", "📤 Export"]
    )

    # ---- Smart Minutes / Notes -------------------------------------------------
    with t_min:
        bullet_card("Lecture summary" if is_lecture else "Executive summary", m["summary"], "No summary was generated.")
        if is_lecture:
            with st.container(border=True):
                st.markdown("<div class='mm-section'>Main concepts</div>", unsafe_allow_html=True)
                if m["main_concepts"]:
                    for c in m["main_concepts"]:
                        st.markdown(f"**{c['concept']}**" + (f": {c['explanation']}" if c["explanation"] else ""))
                else:
                    st.caption("No main concepts detected.")
            a, b = st.columns(2)
            with a:
                with st.container(border=True):
                    st.markdown("<div class='mm-section'>Important definitions</div>", unsafe_allow_html=True)
                    if m["definitions"]:
                        for d in m["definitions"]:
                            st.markdown(f"**{d['term']}**" + (f": {d['definition']}" if d["definition"] else ""))
                    else:
                        st.caption("No explicit definitions detected.")
            with b:
                bullet_card("Key takeaways", m["key_takeaways"], "No takeaways detected.")
            a, b = st.columns(2)
            with a:
                bullet_card("Important examples", m["examples"], "No examples detected.")
            with b:
                bullet_card("Topics to revise", m["revision_topics"], "No revision topics flagged.")
        else:
            a, b = st.columns(2)
            with a:
                bullet_card("✅ Decisions made", m["decisions"], "No explicit decisions detected.")
            with b:
                bullet_card("💬 Discussion points", m["discussion_points"], "No discussion points detected.")
            bullet_card("❓ Open issues", m["open_discussions"], "No open issues detected.")
            if m["attendees"]:
                st.caption("Attendees: " + ", ".join(m["attendees"]))

    # ---- Action Items ------------------------------------------------------------
    with t_act:
        counters_box = st.container()
        cards_box = st.container()
        editor_box = st.container()

        with editor_box:
            if len(st.session_state.tasks_base) == 0:
                st.info("No explicit action items detected in this transcript. "
                        + ("For lectures, tasks appear only if the lecturer mentions homework or submissions. "
                           if is_lecture else "") + "You can still add your own.")
                st.button("➕ Add a task manually", on_click=add_blank_task)
            else:
                st.markdown("<div class='mm-section'>✏️ Edit tasks</div>", unsafe_allow_html=True)
                st.caption("Change owners, priorities, dates or status, tick tasks off, or add/delete rows. "
                           "Everything above, the Task Board and all exports update automatically.")
                edited = st.data_editor(
                    st.session_state.tasks_base,
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    key=f"task_editor_{st.session_state.tasks_ver}",
                    column_config={
                        "Done": st.column_config.CheckboxColumn("Done", width="small"),
                        "Task": st.column_config.TextColumn("Task", width="large", required=True),
                        "Assignee": st.column_config.TextColumn("Assigned to"),
                        "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITIES, required=True),
                        "Status": st.column_config.SelectboxColumn("Status", options=STATUSES, required=True),
                        "Deadline": st.column_config.DateColumn("Deadline", format="DD MMM YYYY"),
                        "Evidence": st.column_config.TextColumn("Evidence", disabled=True,
                                                                help="The exact words from the transcript that support this task."),
                        "Implied Deadline": st.column_config.TextColumn("Said in transcript", disabled=True,
                                                                        help="The original phrase, so you can verify the date the AI chose."),
                    },
                )
                st.session_state.tasks = clean_tasks(edited)

        tdf = st.session_state.tasks
        stats = task_stats(tdf)
        with counters_box:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total tasks", stats["total"])
            c2.metric("Completed", stats["done"])
            c3.metric("In progress", stats["prog"])
            c4.metric("Overdue", stats["overdue"])
        with cards_box:
            valid = tdf[tdf["Task"].astype(str).str.strip() != ""].copy()
            if not valid.empty:
                valid["_p"] = valid["Priority"].map(PRIORITY_RANK)
                valid["_d"] = valid["Deadline"].fillna(pd.Timestamp("2100-01-01"))
                valid = valid.sort_values(["Done", "_p", "_d"], kind="stable")
                st.markdown("<div class='mm-section'>Your action items</div>", unsafe_allow_html=True)
                st.markdown("".join(task_card_html(r) for _, r in valid.iterrows()), unsafe_allow_html=True)
                st.caption("Dates marked “said” show the original wording from the transcript. "
                           "If the AI's date looks wrong, correct it in the table below.")

    # ---- Task Board ----------------------------------------------------------------
    with t_board:
        st.caption("A live board built from your task table. Change a task's status or tick it off in the "
                   "**Action Items** tab and it moves columns here.")
        if st.session_state.tasks["Task"].astype(str).str.strip().eq("").all():
            st.info("No tasks to show yet.")
        else:
            render_kanban(st.session_state.tasks)

    # ---- Transcript ------------------------------------------------------------------
    with t_tr:
        t_html, speakers = render_transcript_html(st.session_state.transcript)
        st.markdown(t_html, unsafe_allow_html=True)
        words = len(st.session_state.transcript.split())
        if speakers:
            st.caption(f"{words} words · Speaker names shown are the labels that were present in the transcript "
                       "(MeetMind does not detect speakers itself).")
        else:
            st.caption(f"{words} words · No speaker labels are shown because none were present in the transcript "
                       "(MeetMind does not perform speaker identification).")

    # ---- Export ------------------------------------------------------------------------
    with t_exp:
        st.caption("Exports always include your latest edits from the task table.")
        df_exp = st.session_state.tasks
        md_text = build_markdown(m, df_exp)
        pdf_bytes = build_pdf(m, df_exp)
        ics_text, ics_n = build_ics(df_exp, m["title"])
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", m["title"]).strip("_") or "meetmind_notes"

        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("<div class='exp'><div class='h'>📝 Markdown</div>"
                        "<div class='d'>Shareable meeting notes you can paste into Notion, GitHub or chat.</div></div>",
                        unsafe_allow_html=True)
            st.download_button("⬇️ Download .md", md_text, f"{safe}.md", "text/markdown", use_container_width=True)
        with e2:
            st.markdown("<div class='exp'><div class='h'>📄 PDF</div>"
                        "<div class='d'>Printable minutes that are easy to hand out or archive.</div></div>",
                        unsafe_allow_html=True)
            if pdf_bytes:
                st.download_button("⬇️ Download .pdf", pdf_bytes, f"{safe}.pdf", "application/pdf", use_container_width=True)
            else:
                st.button("⬇️ Download .pdf", disabled=True, use_container_width=True)
                st.caption("PDF export needs the `fpdf2` package (`pip install fpdf2`).")
        with e3:
            st.markdown("<div class='exp'><div class='h'>📅 Calendar (.ics)</div>"
                        "<div class='d'>Add task deadlines to Google Calendar, Outlook or Apple Calendar.</div></div>",
                        unsafe_allow_html=True)
            st.download_button("⬇️ Download .ics", ics_text, f"{safe}.ics", "text/calendar",
                               use_container_width=True, disabled=ics_n == 0)
            st.caption(f"{ics_n} task(s) with a deadline will become all-day events." if ics_n
                       else "No tasks have a deadline yet. Add dates in the Action Items tab to enable this.")
        with st.expander("Preview Markdown"):
            st.markdown(md_text)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='mm-footer'>MeetMind • AI-powered meeting intelligence<br>Built for BBIT Hackathon 2026</div>",
    unsafe_allow_html=True,
)
