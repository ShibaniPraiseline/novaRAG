# ================= IMPORTS =================
import os
import uuid
import faiss
import whisper
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image
import fitz  # PyMuPDF
import docx2txt
import pytesseract

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= INIT =================
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
whisper_model = whisper.load_model("base")

dimension = 384
index = faiss.IndexFlatL2(dimension)
metadata_store = []

SYSTEM_PROMPT = """
You are a strict document assistant.

Rules:
1) Answer ONLY from the provided context
2) If answer not found → say:
   "The answer is not present in the provided sources."
3) Do NOT use outside knowledge
"""

# ================= HELPERS =================

def embed(text: str):
    return embedder.encode([text])[0]


def add_to_index(text: str, filename: str, page=None):
    vec = embed(text)
    index.add(np.array([vec]).astype("float32"))

    metadata_store.append({
        "id": str(uuid.uuid4()),
        "text": text,
        "path": filename,
        "page": page
    })


def search(query: str, k: int = 5):
    if index.ntotal == 0:
        return []

    qvec = embed(query)
    distances, labels = index.search(np.array([qvec]).astype("float32"), k)

    results = []
    for i in labels[0]:
        if 0 <= i < len(metadata_store):
            item = metadata_store[i]
            results.append({
                "snippet": item["text"],
                "path": item["path"],
                "page": item["page"]
            })
    return results


def build_context(docs):
    parts = []
    for d in docs:
        src = f"[{d.get('path','Unknown')}]"
        if d.get("page"):
            src += f" (page {d['page']})"
        parts.append(f"{src} {d.get('snippet','')}")
    return "\n\n".join(parts)


def build_prompt(context: str, question: str):
    return f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# 🔥 FIXED LLM OUTPUT NORMALIZER
def call_llm(prompt: str) -> str:
    output = llm(prompt)

    if isinstance(output, list):
        first = output[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"])
        if isinstance(first, str):
            return first

    if isinstance(output, dict):
        return str(output.get("generated_text", ""))

    return str(output)


def clean_answer(ans: str):
    return ans.strip()


def grounded_check(answer: str, context: str):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = len(answer_words & context_words) / max(len(answer_words), 1)

    if overlap < 0.30:
        return "The answer is not present in the provided sources."

    return answer


# ================= INGEST =================

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):

    if not file.filename:
        return {"status": "Invalid file"}

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    ingested_chunks = 0
    ext = file.filename.lower()

    # -------- PDF --------
    if ext.endswith(".pdf"):
        doc = fitz.open(save_path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text")
            for chunk in text.split("\n"):
                if chunk.strip():
                    add_to_index(chunk.strip(), file.filename, page_index + 1)
                    ingested_chunks += 1

    # -------- DOCX --------
    elif ext.endswith(".docx"):
        text = docx2txt.process(save_path) or ""
        for chunk in text.split("\n"):
            if chunk.strip():
                add_to_index(chunk.strip(), file.filename)
                ingested_chunks += 1

    # -------- IMAGE --------
    elif ext.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(save_path)
        text = pytesseract.image_to_string(img) or ""
        for chunk in text.split("\n"):
            if chunk.strip():
                add_to_index(chunk.strip(), file.filename)
                ingested_chunks += 1

    # -------- AUDIO --------
    elif ext.endswith((".mp3", ".wav", ".m4a")):
        result = whisper_model.transcribe(save_path)
        text = str(result.get("text", ""))
        for chunk in text.split("."):
            if chunk.strip():
                add_to_index(chunk.strip(), file.filename)
                ingested_chunks += 1

    else:
        return {"status": "Unsupported file"}

    return {"status": "File indexed successfully", "ingested": ingested_chunks}


# ================= QUERY =================

@app.post("/query")
async def query(q: str = Form(...)):

    docs = search(q)

    if not docs:
        return {
            "answer": "The answer is not present in the provided sources.",
            "citations": []
        }

    context = build_context(docs)
    prompt = build_prompt(context, q)

    raw_answer = call_llm(prompt)
    cleaned = clean_answer(raw_answer)
    answer = grounded_check(cleaned, context)

    return {
        "answer": answer,
        "citations": docs
    }
