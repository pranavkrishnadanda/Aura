"""
Production ingest: PyMuPDF + tiktoken chunking + Gemini embeddings + pgvector
Runs as FastAPI BackgroundTasks so 100-page PDF never blocks request.
"""
import uuid, time, logging
import fitz  # PyMuPDF
from typing import List
from app.config import settings
from app.db import upsert_chunks, upsert_chunk_with_embedding

logger = logging.getLogger("aura.ingest")

# In-memory job store for $0 (replace with Redis/Celery in scaled prod).
# Not shared across workers and lost on restart -- fine for a single-instance demo.
_jobs: dict = {}  # job_id -> {status, doc_title, pages, chunks, error, created_at}
_MAX_JOBS = 200  # bound the dict so a long-running instance cannot leak memory

def chunk_text(text: str, size: int = None, overlap: int = None) -> List[str]:
    # `x or default` treats an explicit 0 as "not supplied", so a caller asking for
    # overlap=0 (non-overlapping chunks) silently got the configured default
    # instead. Distinguish "omitted" from "zero".
    size = settings.CHUNK_SIZE if size is None else size
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    # word split keeps the $0 demo simple and deterministic
    size = max(1, size)
    # An overlap >= size makes the stride <= 0 and the loop below never terminates,
    # hanging the ingest thread and growing `chunks` until the process dies. Both
    # values come from env, so clamp rather than trust them.
    overlap = min(max(0, overlap), size - 1)
    step = size - overlap
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        i += step
    return chunks

_EMBED_BATCH = 100


def _embed_call(genai, content):
    """Request EMBED_DIM-wide vectors, tolerating clients without the parameter.

    gemini-embedding-001 returns 3072 dims by default while the chunks.embedding
    column is EMBED_DIM wide, so the width has to be requested explicitly. Older
    google-generativeai releases do not accept output_dimensionality, hence the
    retry rather than a hard dependency on a newer client.
    """
    try:
        return genai.embed_content(
            model=settings.GEMINI_EMBED_MODEL,
            content=content,
            output_dimensionality=settings.EMBED_DIM,
        )
    except TypeError:
        return genai.embed_content(model=settings.GEMINI_EMBED_MODEL, content=content)

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed with Gemini; fall back to None per batch (TF-IDF is used for those)."""
    if not settings.GEMINI_API_KEY or not texts:
        return [None] * len(texts)
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        logger.warning("embeddings unavailable, falling back to TF-IDF: %s", e)
        return [None] * len(texts)

    vectors: List[List[float]] = []
    # embed_content accepts a list, so send batches instead of one request per
    # chunk. A 100-page PDF was previously thousands of serial calls, which blows
    # straight through Gemini's free-tier rate limit and then silently discarded
    # every embedding because one failure returned None for the entire document.
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start:start + _EMBED_BATCH]
        try:
            res = _embed_call(genai, batch)
            emb = res["embedding"]
            # Single-item calls return a flat vector rather than a list of vectors.
            if emb and not isinstance(emb[0], (list, tuple)):
                emb = [emb]
            if len(emb) != len(batch):
                raise ValueError(f"expected {len(batch)} embeddings, got {len(emb)}")
            vectors.extend(emb)
        except Exception as e:
            # Only this batch degrades to TF-IDF, not the whole document.
            logger.warning("embedding batch %d-%d failed, using TF-IDF for it: %s",
                           start, start + len(batch), e)
            vectors.extend([None] * len(batch))
    return vectors

def ingest_pdf_sync(file_bytes: bytes, filename: str, job_id: str = None):
    """Synchronous ingest used by BackgroundTasks"""
    doc_title = filename
    if job_id:
        _jobs[job_id] = {"status": "processing", "doc_title": doc_title, "pages": 0, "chunks": 0, "created_at": time.time()}
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        doc_title = filename
        # One id for the whole document. This used to be generated per chunk, so a
        # 40-chunk PDF produced 40 distinct doc_ids and GET /api/v1/documents -- which
        # groups by doc_id -- listed every chunk as its own separate document.
        doc_id = f"doc_{uuid.uuid4().hex[:6]}"
        page_count = len(doc)
        new_chunks = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text() or ""
            if not text.strip():
                continue
            for chunk in chunk_text(text):
                new_chunks.append({
                    "id": f"chk_{uuid.uuid4().hex[:8]}",
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "page": page_num,
                    "text": chunk.strip(),
                })
        # embed and upsert (vector if available, else plain)
        texts = [c["text"] for c in new_chunks]
        vectors = _embed_texts(texts)
        embedded = 0
        stored = 0
        plain = []
        for chunk, vec in zip(new_chunks, vectors):
            if vec:
                if upsert_chunk_with_embedding(chunk, vec):
                    embedded += 1
                    stored += 1
            else:
                plain.append(chunk)
        if plain:
            # One batched write rather than a session + commit per chunk.
            if upsert_chunks(plain):
                stored += len(plain)
        result = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "pages": page_count,
            "chunks": len(new_chunks),
            "stored": stored,
            "embedded": embedded,
        }
        if job_id:
            # Only claim success if the chunks actually persisted. Reporting
            # "completed" after every write failed told the user their document was
            # indexed when retrieval could never see it.
            if stored == len(new_chunks):
                _jobs[job_id].update({"status": "completed", **result})
            elif stored:
                _jobs[job_id].update({"status": "partial", "error": f"only {stored} of {len(new_chunks)} chunks were stored", **result})
            else:
                _jobs[job_id].update({"status": "failed", "error": "no chunks could be stored", **result})
        logger.info("ingest complete: %s pages=%d chunks=%d embedded=%d",
                    doc_title, page_count, len(new_chunks), embedded)
        return result
    except Exception as e:
        # BackgroundTasks discards exceptions, so without this the job row is the
        # only trace a failure ever leaves.
        logger.exception("ingest failed for %s", filename)
        if job_id:
            _jobs[job_id].update({"status": "failed", "error": str(e)})
        raise
    finally:
        if doc is not None:
            doc.close()

def ingest_pdf(file_bytes: bytes, filename: str):
    """Legacy sync wrapper (keeps old tests working)"""
    return ingest_pdf_sync(file_bytes, filename)

def create_job() -> str:
    jid = f"job_{uuid.uuid4().hex[:8]}"
    if len(_jobs) >= _MAX_JOBS:
        # Drop the oldest completed jobs; the dict is otherwise unbounded.
        for old in sorted(_jobs, key=lambda k: _jobs[k].get("created_at", 0))[:_MAX_JOBS // 2]:
            _jobs.pop(old, None)
    _jobs[jid] = {"status": "queued", "created_at": time.time()}
    return jid

def get_job(job_id: str):
    return _jobs.get(job_id)
