# Local RAG Explorer

A fully local Retrieval-Augmented Generation (RAG) application powered by Ollama, FastAPI, and a custom frontend interface.

This project allows users to upload documents, generate embeddings locally, retrieve relevant context using semantic search, and chat with documents using local LLMs such as Llama3, Phi3, and TinyLlama.

---

# Features

- Fully local AI stack
- Ollama-powered LLM inference
- Semantic document retrieval
- Local embeddings using `nomic-embed-text`
- FastAPI backend
- Custom lightweight frontend
- Multiple model support
- Similarity search
- Chunk-based retrieval
- Upload `.txt`, `.pdf`, and `.docx` files
- Real-time RAG metrics
- No external APIs required

---

# Tech Stack

## Backend
- FastAPI
- Python
- NumPy
- Ollama

## Frontend
- HTML
- CSS
- Vanilla JavaScript

## Models
- llama3
- phi3
- tinyllama
- nomic-embed-text

---

# Architecture

```text
Frontend
   ↓
FastAPI Backend
   ↓
Embedding Generation
   ↓
Semantic Retrieval
   ↓
Ollama LLM
   ↓
Response Generation
```

---

# Project Structure

```text
backend/
│
├── app/
│   ├── main.py
│   ├── uploads/
│   └── __pycache__/
│
├── frontend/
│   └── index.html
│
├── chroma_db/
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/local-rag-explorer.git

cd local-rag-explorer
```

---

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download Ollama:

:contentReference[oaicite:0]{index=0}

---

# Pull Models

```bash
ollama pull llama3

ollama pull phi3

ollama pull tinyllama

ollama pull nomic-embed-text
```

---

# Run Backend

Navigate to backend app directory:

```bash
cd backend/app
```

Run FastAPI server:

```bash
uvicorn main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

# Run Frontend

Open:

```text
frontend/index.html
```

OR use VS Code Live Server.

Frontend runs on:

```text
http://127.0.0.1:5500
```

---

# Supported File Types

- TXT
- PDF
- DOCX

---

# How It Works

## 1. Upload Document
Documents are uploaded to the backend and stored locally.

## 2. Chunking
Text is split into semantic chunks.

## 3. Embedding Generation
Chunks are converted into vector embeddings using:

```text
nomic-embed-text
```

## 4. Retrieval
Relevant chunks are retrieved using cosine similarity.

## 5. Response Generation
Retrieved context is sent to a local LLM through Ollama.

---

# Available Models

| Model | Speed | Quality |
|---|---|---|
| tinyllama | Fast | Basic |
| phi3 | Medium | Good |
| llama3 | Slow | Best |

---

# Example Queries

```text
Summarize the process

What to do in batch out balance error?

Explain the retry workflow

What is the validation process?
```

---

# Performance Optimizations

- Query embedding caching
- Reduced chunk context
- Top-k retrieval tuning
- TinyLlama support for fast inference
- Optimized prompt engineering

---

# Future Improvements

- Streaming responses
- ChromaDB persistent storage
- FAISS vector search
- Hybrid retrieval
- Dark mode
- Drag-and-drop upload
- Conversation memory
- Markdown rendering
- Authentication
- Docker support

---

# Screenshots

## Main Interface

Add screenshots here.

---

# License

MIT License

---

# Acknowledgements

- Ollama
- FastAPI
- Meta Llama
- Microsoft Phi
- TinyLlama
- ChromaDB

---

# Author

Sourabh Kumar Singh
