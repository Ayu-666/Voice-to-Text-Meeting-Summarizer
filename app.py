import streamlit as st
from groq import Groq

st.title("Groq Connection Test")

api_key = st.text_input("Groq API Key", type="password")

if st.button("Test Groq"):

    if not api_key:
        st.error("Enter your API key")
        st.stop()

    try:
        client = Groq(api_key=api_key)

        models = client.models.list()

        st.success("✅ API key works!")

        st.write("Models available to this API key:")

        for model in models.data:
            st.code(model.id)

    except Exception as e:
        st.error("❌ Groq connection failed")
        st.code(str(e))
