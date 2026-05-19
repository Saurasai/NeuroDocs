# backend/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import uuid
import fitz
from docx import Document
import os
import math

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MEMORY STORE
# ─────────────────────────────────────────────────────────────

documents = []
chunks_store = []

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def chunk_text(text, chunk_size=4):

    sentences = text.split(".")

    chunks = []

    for i in range(0, len(sentences), chunk_size):

        chunk = ".".join(
            sentences[i:i + chunk_size]
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )

    data = response.json()

    return data["embedding"]


def extract_text(filepath):
    ext = filepath.split(".")[-1].lower()

    if ext == "txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == "pdf":
        doc = fitz.open(filepath)

        text = ""

        for page in doc:
            text += page.get_text()

        return text

    elif ext == "docx":
        doc = Document(filepath)

        return "\n".join(
            [p.text for p in doc.paragraphs]
        )

    else:
        raise Exception("Unsupported file format")


def retrieve_chunks(query, top_k=3):
    query_embedding = get_embedding(query)

    scored = []

    for chunk in chunks_store:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        scored.append({
            "score": score,
            **chunk
        })

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored[:top_k]


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    model: str = "llama3"


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "llm": LLM_MODEL,
        "embed_model": EMBED_MODEL
    }


@app.get("/health")
def health():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags"
        )

        return {
            "ollama": "online",
            "models": response.json()
        }

    except Exception as e:
        return {
            "ollama": "offline",
            "error": str(e)
        }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    filepath = os.path.join(
        UPLOAD_DIR,
        f"{file_id}_{file.filename}"
    )

    with open(filepath, "wb") as f:
        f.write(await file.read())

    text = extract_text(filepath)

    doc = {
        "id": file_id,
        "name": file.filename,
        "text": text
    }

    documents.append(doc)

    # chunk + embed
    chunks = chunk_text(text)


    

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        chunks_store.append({
            "doc_id": file_id,
            "doc_name": file.filename,
            "chunk_id": idx,
            "text": chunk,
            "embedding": embedding
        })

    return {
        "message": "Document uploaded successfully",
        "document": file.filename,
        "chunks_created": len(chunks)
    }


@app.get("/documents")
def get_documents():
    return [
        {
            "id": d["id"],
            "name": d["name"]
        }
        for d in documents
    ]


@app.post("/search")
def search(req: ChatRequest):
    results = retrieve_chunks(
        req.query,
        req.top_k
    )

    return {
        "query": req.query,
        "results": [
            {
                "score": round(r["score"], 4),
                "source": r["doc_name"],
                "text": r["text"]
            }
            for r in results
        ]
    }


@app.post("/chat")
def chat(req: ChatRequest):

    try:

        import time

        start_time = time.time()

        print("CHAT STARTED")

        retrieved = retrieve_chunks(
            req.query,
            req.top_k
        )

        retrieval_time = round(
            time.time() - start_time,
            2
        )

        print("RETRIEVED")

        context = "\n\n---\n\n".join([
            f"[Source: {r['doc_name']}]\n{r['text']}"
            for r in retrieved
        ])

        system_prompt = f"""
You are an enterprise RAG assistant.

STRICT RULES:
1. Answer ONLY using the retrieved context.
2. Do NOT add assumptions.
3. Do NOT explain unrelated procedures.
4. If exact answer is unavailable, say:
   "I could not find the exact procedure."
5. Summarize only the directly relevant steps.
6. Keep answers short and structured.

Retrieved Context:
{context}
"""

        llm_start = time.time()

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": req.model,
                "prompt": f"{system_prompt}\n\nQuestion:\n{req.query}",
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=120
        )

        data = response.json()

        generation_time = round(
            time.time() - llm_start,
            2
        )

        total_time = round(
            time.time() - start_time,
            2
        )

        return {

            "answer": data["response"],

            "metrics": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "chunks_used": len(retrieved)
            },

            "sources": [
                {
                    "document": r["doc_name"],
                    "score": round(r["score"], 4),
                    "text": r["text"][:500]
                }
                for r in retrieved
            ]
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "answer": f"Backend error: {str(e)}",
            "sources": [],
            "metrics": {}
        }