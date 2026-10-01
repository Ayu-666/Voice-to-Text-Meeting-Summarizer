# Hackathon Progress & Technical Hurdles

## Phase 1: Core Integration
- Successfully linked Streamlit's native `st.audio_input` to the Groq Whisper API. 
- Mapped out the JSON schema for the LLM to extract the Executive Summary, Decisions, Open Discussions, and Tasks[cite: 11].

## Phase 2: The "Vanishing Button" State Bug
- **Challenge:** In Streamlit, interacting with elements inside a nested loop or `if` statement can cause the page to refresh and lose the session state. Our initial UI was dropping the audio payload before the AI could process it.
- **Solution:** Restructured the app into a linear execution flow. Pushed the final extracted data into a global `st.session_state.result` dictionary. This ensured the Action-Item Matrix remained stable and editable even when the UI refreshed.

## Phase 3: AI Rate Limit Mitigation
- **Challenge:** We initially used the `llama-3.3-70b-versatile` model. During testing, the heavy parameter load triggered 404/rate-limit errors on the free-tier API, threatening the live demo.
- **Solution:** Engineered a lightweight, robust prompt and pivoted the LLM engine to `openai/gpt-oss-20b`. This model is exceptionally fast, highly capable of structured JSON generation, and immune to the heavy rate limits of the 70B models. Added a strict regex fallback to parse JSON even if the AI hallucinates markdown formatting.

## Phase 4: Export Engine
- Implemented `fpdf2` for clean PDF generation (FR-6)[cite: 11].
- Wrote a custom Python script to dynamically generate `.ics` calendar payloads by parsing the Pandas dataframe for valid deadline dates (FR-6)[cite: 11].
