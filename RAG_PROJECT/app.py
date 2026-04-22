import streamlit as st
from main import app

st.set_page_config(page_title="Customer Support Bot")

st.title("🤖 Customer Support Assistant")

query = st.text_input("Ask your question:")

if st.button("Submit"):
    if query:
        result = app.invoke({
            "query": query,
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "escalate": False
        })

        st.success(result["answer"])
    else:
        st.warning("Please enter a question")