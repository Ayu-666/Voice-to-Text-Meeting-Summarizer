# Planning & Architecture Decisions

## 🎯 Target Personas
Primary Users: Student club leads, hackathon team members, and class representatives who need to capture fast-moving discussions without acting as full-time stenographers[cite: 11].

## 🏗️ Architectural Pivot & Stack Defense
The hackathon handbook recommended a decoupled architecture using Next.js (Frontend), FastAPI (Backend), and a split AI strategy (Groq + Gemini)[cite: 11]. During our initial planning phase, we evaluated this stack but actively chose to pivot to a **Streamlit Python Monolith powered exclusively by Groq**.

**Why we made this decision:**
1. **Latency & The 30-Second Test:** The rubric demands high technical implementation where judges can speak into a microphone for 30 seconds and see minutes generated live[cite: 11]. Network hops between a Vercel frontend, Render backend, and multiple AI APIs create dangerous latency. Streamlit consolidates the UI and backend into a single thread, executing the Whisper call instantly.
2. **Interactive Task Board (FR-5):** Building an editable, tabular matrix in React requires heavy state management[cite: 11]. By using Python, we leveraged `pandas` and Streamlit's native `st.data_editor`, providing a flawless, memory-safe data grid in just two lines of code.
3. **Consolidated AI (Groq):** Relying on Groq for both Whisper and the LLM (`openai/gpt-oss-20b`) eliminates the risk of a secondary API (like Gemini) timing out or failing during the live demonstration.
