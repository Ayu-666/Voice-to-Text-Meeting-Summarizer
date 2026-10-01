# Deployment Guide

## Overview
MeetMind is designed to run in any Python environment with zero complex server configurations. The application can be hosted instantly on Streamlit Community Cloud or Replit.

## Dependencies (`requirements.txt`)
- `streamlit>=1.40.0`: UI Framework and State Management
- `pandas`: Task Matrix Data Handling
- `groq`: AI Inference (Whisper & Text Extraction)
- `fpdf2`: PDF Export Generation

## Environment Variables
The application does not hardcode sensitive keys. It requires:
- `GROQ_API_KEY`: Can be provided dynamically via the UI on launch, or stored in Streamlit Secrets / Replit Environment Variables.

## Deployment Steps (Streamlit Cloud)
1. Push the repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Click **New App** and select the GitHub repository.
4. Set the Main file path to `app.py`.
5. Click **Deploy**.

## Deployment Steps (Replit)
1. Create a new Streamlit Repl.
2. Upload the source files.
3. Add `GROQ_API_KEY` in the Secrets tab.
4. Press **Run**. The web view will initialize automatically via port 5000.
