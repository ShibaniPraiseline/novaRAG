# novaRAG_app.py
import streamlit as st
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NovaRAG Offline Multimodal RAG", layout="wide")
st.title("NovaRAG Offline Multimodal RAG")

# session memory
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# 1️⃣ File Ingestion
# --------------------------
st.header("1️⃣ Ingest File")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX, IMAGE, or AUDIO",
    type=["pdf", "docx", "png", "jpg", "jpeg", "wav", "mp3", "m4a"],
    help="Limit 200MB per file"
)

if uploaded_file:
    st.info(f"Selected file: {uploaded_file.name} ({uploaded_file.size/1024/1024:.2f} MB)")

    if st.button("Ingest File"):
        with st.spinner("Uploading and ingesting..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue())
                }

                resp = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
                resp.raise_for_status()

                data = resp.json()
                st.success("✅ File ingested successfully!")

                if "chunks" in data:
                    st.caption(f"Indexed chunks: {data['chunks']}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Start FastAPI first.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --------------------------
# 2️⃣ Chat Interface
# --------------------------
st.header("2️⃣ Ask Questions")

question = st.chat_input("Ask anything about your documents...")

if question:
    st.session_state.history.append(("user", question))

    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"q": question},
                timeout=180
            )

            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "No answer returned.")
            citations = data.get("citations", [])

            st.session_state.history.append(("assistant", answer, citations))

        except Exception as e:
            st.session_state.history.append(("assistant", f"Error: {e}", []))

# --------------------------
# Conversation Display
# --------------------------
for item in st.session_state.history:
    if item[0] == "user":
        with st.chat_message("user"):
            st.markdown(item[1])

    else:
        with st.chat_message("assistant"):
            st.markdown(item[1])

            citations = item[2]
            if citations:
                st.markdown("**Sources:**")
                for c in citations:
                    file_name = Path(c.get("path", "Unknown")).name
                    page_no = c.get("page", "N/A")
                    snippet = c.get("snippet", "")

                    with st.expander(f"{file_name} (page {page_no})"):
                        st.write(snippet)
