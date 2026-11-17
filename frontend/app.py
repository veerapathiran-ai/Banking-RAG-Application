import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Bedrock Nova Pro RAG Assistant", layout="wide")

st.title("🏦 Enterprise RAG Assistant (Amazon Nova Pro + AWS S3)")

# Sidebar for Document Management
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF to AWS S3", type=["pdf"])
    
    if uploaded_file and st.button("🚀 Upload to AWS S3"):
        with st.spinner("Uploading to S3..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload-to-s3", files=files)
                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error(res.json().get("detail", "Upload failed"))
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.markdown("---")
    if st.button("📥 Load & Index from AWS S3"):
        with st.spinner("Syncing S3 files and building local FAISS index..."):
            try:
                res = requests.post(f"{BACKEND_URL}/sync-s3")
                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error(res.json().get("detail", "Sync failed"))
            except Exception as e:
                st.error(f"Connection error: {e}")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with Amazon Nova Pro..."):
            try:
                payload = {"question": prompt}
                res = requests.post(f"{BACKEND_URL}/ask", json=payload)
                if res.status_code == 200:
                    answer = res.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    err_msg = res.json().get("detail", "Unknown error occurred")
                    st.error(err_msg)
            except Exception as e:
                st.error(f"Connection error: {e}")