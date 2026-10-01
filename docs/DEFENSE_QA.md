# Judge Defense & Q/A

**Q: The handbook recommended Next.js and FastAPI. Why did you use Streamlit?**[cite: 11]
**A:** We analyzed the rubric and realized the highest priority was the "30-second live mic test" and the "Interactive Task Board" (FR-5)[cite: 11]. Next.js/FastAPI introduces network latency between the frontend and backend. Streamlit is a unified monolith, meaning the microphone audio goes straight to the backend memory. Furthermore, Streamlit's native `pandas` integration allowed us to build a more robust, memory-safe editable task matrix than we could have built from scratch in React given the time constraints. The handbook explicitly states teams have full technical freedom[cite: 11], and this stack allowed us to execute perfectly.

**Q: How are you ensuring data privacy for these meetings?**[cite: 11]
**A:** Our application does not utilize any disk I/O. The audio recorded from the microphone is held temporarily in RAM as a byte-stream. We explicitly coded a garbage collection command (`del audio_bytes`) that executes the millisecond the Groq Whisper API returns the transcript. Nothing is saved, and nothing can be leaked.

**Q: What happens if the AI fails to generate the required JSON structure?**
**A:** We implemented two layers of defense. First, we use a highly rigid prompt instructing the model (`gpt-oss-20b`) to output raw JSON. Second, we wrapped the response in a Regex extractor. If the AI hallucinates markdown (e.g., ` ```json `), our code strips the formatting, finds the first `{` and last `}`, and parses the payload safely so the app never crashes on the user.

**Q: How does the calendar export work?**[cite: 11]
**A:** When the AI extracts a task, it looks for an implied deadline. If one is found (or if the user adds one manually via the Task Board), our custom `build_ics` Python function translates that row into a standard VEVENT format. When the user clicks download, it generates an `.ics` file that automatically maps the Task Description, Assigned Person, and Priority directly into Google Calendar or Apple Calendar.
