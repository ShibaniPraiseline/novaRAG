"""
Offline Multimodal RAG - FastAPI Prototype with Ollama Llama-3
"""

import os
import io
import uuid
import json
import sqlite3
from typing import List
from pathlib import Path

import numpy as np
import faiss
import torch

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from docx import Document
import whisper

# Ollama (local LLM)
from ollama import chat


# ================= CONFIG =================
DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)

INDEX_FILE = DATA_DIR / "faiss_index.bin"
META_DB = DATA_DIR / "metadata.db"

EMBED_DIM = 512
CHUNK_SIZE = 800
OVERLAP = 120
TOP_K = 8


# ================= LOAD MODELS =================
print("Loading models...")

text_encoder = SentenceTransformer("all-mpnet-base-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

whisper_model = whisper.load_model("small")

text_dim = text_encoder.get_sentence_embedding_dimension()
text_proj = torch.nn.Linear(text_dim, EMBED_DIM) if text_dim != EMBED_DIM else None

clip_text_dim = clip_model.text_model.config.hidden_size
clip_proj = torch.nn.Linear(clip_text_dim, EMBED_DIM) if clip_text_dim != EMBED_DIM else None


# ================= FAISS =================
if INDEX_FILE.exists():
    index = faiss.read_index(str(INDEX_FILE))
else:
    quantizer = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, 100, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 10


# ================= METADATA DB =================
conn = sqlite3.connect(META_DB, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS docs (
    id TEXT PRIMARY KEY,
    source_path TEXT,
    source_type TEXT,
    snippet TEXT,
    page INTEGER,
    start_sec REAL,
    end_sec REAL,
    extra JSON
)
""")
conn.commit()


# ================= UTILS =================
def chunk_text(text):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - OVERLAP
    return chunks


def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def embed_text_list(texts: List[str]) -> np.ndarray:
    emb = text_encoder.encode(texts, convert_to_numpy=True)
    emb = torch.tensor(emb)
    if text_proj:
        emb = text_proj(emb)
    return normalize(emb.detach().cpu().numpy())


def embed_image(data) -> np.ndarray:
    if isinstance(data, bytes):
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(data).convert("RGB")

    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)

    if clip_proj:
        emb = clip_proj(emb)

    return normalize(emb.detach().cpu().numpy())


def save_index():
    faiss.write_index(index, str(INDEX_FILE))


def upsert_meta(uid, path, stype, snippet=None, page=None, s=None, e=None):
    cur.execute("""
    INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?)
    """, (uid, path, stype, snippet, page, s, e, json.dumps({})))
    conn.commit()


# ================= INGEST =================
def ingest_pdf(data, name):
    path = DATA_DIR / f"{uuid.uuid4()}_{name}"
    path.write_bytes(data)

    doc = fitz.open(path)
    vectors = []

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text()
        for chunk in chunk_text(text):
            uid = str(uuid.uuid4())
            emb = embed_text_list([chunk])[0]
            vectors.append(emb)
            upsert_meta(uid, str(path), "pdf", chunk[:300], pno + 1)

    if vectors:
        arr = np.array(vectors).astype("float32")
        if not index.is_trained:
            index.train(arr)
        index.add(arr)
        save_index()

    return {"status": "PDF ingested"}


def ingest_docx(data, name):
    path = DATA_DIR / f"{uuid.uuid4()}_{name}"
    path.write_bytes(data)

    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)

    vectors = []
    for chunk in chunk_text(text):
        uid = str(uuid.uuid4())
        emb = embed_text_list([chunk])[0]
        vectors.append(emb)
        upsert_meta(uid, str(path), "docx", chunk[:300])

    if vectors:
        arr = np.array(vectors).astype("float32")
        if not index.is_trained:
            index.train(arr)
        index.add(arr)
        save_index()

    return {"status": "DOCX ingested"}


def ingest_image(data, name):
    path = DATA_DIR / f"{uuid.uuid4()}_{name}"
    path.write_bytes(data)

    ocr = pytesseract.image_to_string(Image.open(path))
    vectors = []

    if ocr.strip():
        for chunk in chunk_text(ocr):
            uid = str(uuid.uuid4())
            emb = embed_text_list([chunk])[0]
            vectors.append(emb)
            upsert_meta(uid, str(path), "image_ocr", chunk[:300])

    img_emb = embed_image(path)[0]
    vectors.append(img_emb)
    upsert_meta(str(uuid.uuid4()), str(path), "image")

    arr = np.array(vectors).astype("float32")
    if not index.is_trained:
        index.train(arr)
    index.add(arr)
    save_index()

    return {"status": "Image ingested"}


def ingest_audio(data, name):
    path = DATA_DIR / f"{uuid.uuid4()}_{name}"
    path.write_bytes(data)

    res = whisper_model.transcribe(str(path))
    vectors = []

    for seg in res["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        uid = str(uuid.uuid4())
        emb = embed_text_list([text])[0]
        vectors.append(emb)
        upsert_meta(uid, str(path), "audio", text[:300], None, seg["start"], seg["end"])

    if vectors:
        arr = np.array(vectors).astype("float32")
        if not index.is_trained:
            index.train(arr)
        index.add(arr)
        save_index()

    return {"status": "Audio ingested"}


# ================= QUERY =================
def search(query):
    q_emb = embed_text_list([query])
    D, I = index.search(q_emb.astype("float32"), TOP_K)

    cur.execute("SELECT id FROM docs")
    all_ids = [r[0] for r in cur.fetchall()]

    matched = [all_ids[i] for i in I[0] if i < len(all_ids)]

    results = []
    for mid in matched:
        cur.execute("SELECT * FROM docs WHERE id=?", (mid,))
        row = cur.fetchone()
        if row:
            results.append({
                "path": row[1],
                "type": row[2],
                "snippet": row[3],
                "page": row[4]
            })

    return results


def call_llm(prompt):
    res = chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return res["message"]["content"]


def build_prompt(q, docs):
    ctx = "\n".join(
        f"[{i+1}] {d['snippet']}" for i, d in enumerate(docs)
    )

    return f"""
You are an offline RAG assistant.
Answer ONLY from the sources.
If answer not found, say: I don't know.

Question: {q}

Sources:
{ctx}
"""


# ================= FASTAPI =================
app = FastAPI(title="Offline Multimodal RAG")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    ext = file.filename.lower()

    if ext.endswith(".pdf"):
        return ingest_pdf(data, file.filename)
    if ext.endswith(".docx"):
        return ingest_docx(data, file.filename)
    if ext.endswith((".png", ".jpg", ".jpeg")):
        return ingest_image(data, file.filename)
    if ext.endswith((".wav", ".mp3", ".m4a")):
        return ingest_audio(data, file.filename)

    return JSONResponse({"error": "Unsupported format"}, 400)


@app.post("/query")
async def query(q: str = Form(...)):
    docs = search(q)
    prompt = build_prompt(q, docs)
    answer = call_llm(prompt)
    return {"answer": answer, "sources": docs}


@app.get("/status")
def status():
    return {"vectors": index.ntotal, "trained": index.is_trained}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
