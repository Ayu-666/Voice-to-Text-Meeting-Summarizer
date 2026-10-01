# MeetMind: Smart Meeting & Lecture Summarizer (AI-03)

## 📌 Real-World Context
Student club meetings, project group discussions, and long lectures often conclude without clear accountability, resulting in forgotten action items and missed deadlines[cite: 11]. MeetMind is an automated summarizer that ingests audio recordings or raw transcripts, extracts decisions made, assigns action items to named attendees, and exports them directly to calendar formats[cite: 11].

## 🚀 Core Features (Rubric Alignment)
- **FR-1 (Audio/Text Input):** Upload meeting audio files (MP3/WAV up to 25MB), record via live mic, or paste raw meeting transcripts[cite: 11].
- **FR-2 (Speech-to-Text Transcription):** Fast, accurate transcription utilizing the Groq Whisper API (`whisper-large-v3`)[cite: 11].
- **FR-3 (Structured Meeting Minutes):** AI automatically extracts an Executive 3-bullet summary, key decisions ratified, and open discussions[cite: 11].
- **FR-4 (Action-Item Matrix):** Extracts actionable tasks mapped to Task Description, Assigned Person, Priority, and Implied Deadline[cite: 11].
- **FR-5 (Interactive Task Board):** Inline Kanban/checklist board where users can edit task owners, add dates, and tick off tasks before export[cite: 11].
- **FR-6 (One-Click Export):** Export formatted meeting minutes to Markdown, PDF (via `fpdf2`), or `.ics` calendar invitation files[cite: 11].

## 🛠️ Tech Stack
We deviated from the handbook's Next.js/FastAPI recommendation to optimize for zero-latency live demos and robust data handling[cite: 11].
- **Frontend & Backend:** Streamlit (Python monolith for seamless data state management)
- **Data Handling:** Pandas (Natively powers the Interactive Task Board)
- **AI Processing:** Groq API (Single API for both Speech-to-Text and LLM extraction)
- **Models Used:** `whisper-large-v3` (Audio), `openai/gpt-oss-20b` (Text Extraction)

## 🔒 Non-Functional Requirements
- **Data Privacy Guarantee:** Audio is processed entirely in-memory and explicitly purged (`del audio_bytes`) immediately after transcription[cite: 11]. No files are saved to disk.
- **UI/UX:** Clean typography, progress bars during transcription, and visual toast notifications[cite: 11].

## ⚙️ Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Enter your Groq API key in the UI sidebar to begin.
