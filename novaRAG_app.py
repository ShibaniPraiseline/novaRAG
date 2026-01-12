# novaRAG_app.py
import streamlit as st
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NovaRAG Offline Multimodal RAG", layout="wide")
st.title("NovaRAG Offline Multimodal RAG")

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

                resp = requests.post(f"{API_URL}/ingest", files=files)
                resp.raise_for_status()

                data = resp.json()
                st.success(f"✅ File ingested successfully! Chunks ingested: {data.get('ingested', 0)}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is FastAPI running on port 8000?")
            except requests.exceptions.JSONDecodeError:
                st.error("❌ Backend did not return valid JSON.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --------------------------
# 2️⃣ Ask a Question
# --------------------------
st.header("2️⃣ Ask a Question")

question = st.text_input("Type your question about the ingested documents:")

if st.button("Get Answer") and question.strip():
    with st.spinner("Fetching answer..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data=f"q={question}",
                timeout=120
            )

            resp.raise_for_status()
            data = resp.json()

            st.subheader("💬 Answer")
            st.markdown(data.get("answer", "No answer returned."))

            st.subheader("📄 Citations / Sources")
            for c in data.get("citations", []):
                file_name = Path(c["path"]).name
                page_no = c.get("page", "N/A")
                with st.expander(f"{file_name} (page {page_no})"):
                    st.write(c.get("snippet", ""))

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is FastAPI running?")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Backend error: {e.response.status_code}")
            st.text(e.response.text)
        except Exception as e:
            st.error(f"❌ Error: {e}")
