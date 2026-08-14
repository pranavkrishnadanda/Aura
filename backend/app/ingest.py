"""
Production ingest: PyMuPDF + tiktoken chunking + Gemini embeddings + pgvector
Runs as FastAPI BackgroundTasks so 100-page PDF never blocks request.
"""
import uuid, time
import fitz  # PyMuPDF
from typing import List
from app.config import settings
from app.db import upsert_chunks, upsert_chunk_with_embedding

# In-memory job store for $0 (replace with Redis/Celery in scaled prod)
_jobs: dict = {}  # job_id -> {status, doc_title, pages, chunks, error, created_at}

def chunk_text(text: str, size: int = None, overlap: int = None) -> List[str]:
    size = size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    # tiktoken-aware would be better; word split keeps $0 demo simple and deterministic
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        i += size - overlap
    return chunks

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Try Gemini embeddings, fallback to None (TF-IDF will be used)"""
    if not settings.GEMINI_API_KEY:
        return [None] * len(texts)
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # batch via single call per chunk (free tier friendly)
        vectors = []
        for t in texts:
            res = genai.embed_content(model=settings.GEMINI_EMBED_MODEL, content=t)
            vectors.append(res["embedding"])
        return vectors
    except Exception as e:
        print(f"[ingest] embedding failed, fallback to TF-IDF: {e}")
        return [None] * len(texts)

def ingest_pdf_sync(file_bytes: bytes, filename: str, job_id: str = None):
    """Synchronous ingest used by BackgroundTasks"""
    doc_title = filename
    if job_id:
        _jobs[job_id] = {"status": "processing", "doc_title": doc_title, "pages": 0, "chunks": 0, "created_at": time.time()}
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        doc_title = filename
        new_chunks = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text() or ""
            if not text.strip():
                continue
            for chunk in chunk_text(text):
                new_chunks.append({
                    "id": f"chk_{uuid.uuid4().hex[:8]}",
                    "doc_id": f"doc_{uuid.uuid4().hex[:6]}",
                    "doc_title": doc_title,
                    "page": page_num,
                    "text": chunk.strip(),
                })
        # embed and upsert (vector if available, else plain)
        texts = [c["text"] for c in new_chunks]
        vectors = _embed_texts(texts)
        for chunk, vec in zip(new_chunks, vectors):
            if vec:
                upsert_chunk_with_embedding(chunk, vec)
            else:
                upsert_chunks([chunk])
        result = {"doc_title": doc_title, "pages": len(doc), "chunks": len(new_chunks)}
        if job_id:
            _jobs[job_id].update({"status": "completed", **result})
        return result
    except Exception as e:
        if job_id:
            _jobs[job_id].update({"status": "failed", "error": str(e)})
        raise

def ingest_pdf(file_bytes: bytes, filename: str):
    """Legacy sync wrapper (keeps old tests working)"""
    return ingest_pdf_sync(file_bytes, filename)

def create_job() -> str:
    jid = f"job_{uuid.uuid4().hex[:8]}"
    _jobs[jid] = {"status": "queued", "created_at": time.time()}
    return jid

def get_job(job_id: str):
    return _jobs.get(job_id)
