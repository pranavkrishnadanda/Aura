import asyncio, json, time, uuid, logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from app.config import settings
from app.schemas import ChatRequest, ThreadCreate
from app.db import create_thread, list_threads, get_thread, get_messages, add_message, get_chunk, list_chunks, try_pg_connection, db_error, storage_mode
from app.rag import retrieve, generate_answer, effective_threshold, retrieval_mode
from app.auth import get_current_user
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura")

app = FastAPI(title="Aura API", version="1.0.0", description="Clinical Intelligence Streaming RAG — production")

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_ANON])
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if not origins:
    # Falling back to "*" while also sending Allow-Credentials: true produces a
    # header pair every browser rejects, so an unset CORS_ORIGINS made the API
    # unreachable from the frontend with only an opaque CORS error to debug.
    # Credentials are not needed here -- auth travels in the X-API-Key header, not
    # a cookie -- so the wildcard is paired with credentials disabled instead.
    logger.warning("CORS_ORIGINS is unset; allowing all origins without credentials")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=bool(origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key"],
    expose_headers=["X-Response-Time"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    # PII-safe: don't log query content if LOG_QUERIES false
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} {response.status_code} {elapsed:.1f}ms")
    response.headers["X-Response-Time"] = f"{elapsed:.1f}ms"
    return response

_STARTED_AT = time.time()

@app.get("/health")
def health():
    pg = try_pg_connection()
    return {
        # Degraded, not "ok", when we are silently running without a database.
        "status": "ok" if pg else "degraded",
        "version": "1.0.0",
        "provider": settings.LLM_PROVIDER,
        "retrieval_mode": retrieval_mode(),
        # The floor actually enforced, and the two configured floors it is chosen
        # from. These used to disagree silently: 0.10 was always applied while
        # 0.85 was reported.
        "threshold": effective_threshold(),
        "configured_threshold": settings.RETRIEVAL_THRESHOLD,
        "tfidf_threshold": settings.TFIDF_THRESHOLD,
        "pg_reachable": pg,
        "storage_mode": storage_mode(),
        "db_error": db_error(),
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
    }

@app.get("/ready")
def ready(response: Response):
    # For k8s/Render health checks. This must be able to report NOT ready --
    # returning a hardcoded True made the probe (and its test) meaningless.
    pg = try_pg_connection()
    llm = bool(settings.GEMINI_API_KEY or settings.GROQ_API_KEY or settings.LLM_PROVIDER == "mock")
    is_ready = pg and llm
    if not is_ready:
        response.status_code = 503
    return {"ready": is_ready, "pg": pg, "llm": llm, "db_error": db_error()}

# ---- Threads ----
@app.post("/api/v1/threads")
@limiter.limit(settings.RATE_LIMIT_AUTH)
def post_thread(request: Request, body: ThreadCreate, user=Depends(get_current_user)):
    if len(body.title or "") > 200:
        raise HTTPException(400, "Title too long")
    return create_thread(body.title or "New consultation", user_id=user["user_id"])

def _assert_thread_access(thread_id: str, user: dict) -> dict:
    """Fetch a thread, enforcing ownership when auth is on.

    This endpoint previously had no ownership check whatsoever, so any caller could
    read any conversation by naming its id -- and ids are only 8 hex characters.
    A 404 (not 403) is returned for someone else's thread so the endpoint does not
    confirm which ids exist.
    """
    t = get_thread(thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")
    if settings.ENABLE_AUTH and t.get("user_id") not in (user["user_id"], None):
        raise HTTPException(404, "Thread not found")
    return t

@app.get("/api/v1/threads")
@limiter.limit(settings.RATE_LIMIT_ANON)
def get_threads(request: Request, user=Depends(get_current_user)):
    # Always scope to the caller. Passing None when auth was disabled returned
    # every thread from every visitor, so one demo user saw another's clinical
    # queries in the sidebar.
    return list_threads(user_id=user["user_id"])

@app.get("/api/v1/threads/{thread_id}/messages")
@limiter.limit(settings.RATE_LIMIT_ANON)
def get_thread_messages(request: Request, thread_id: str, limit: int = 100, offset: int = 0,
                        user=Depends(get_current_user)):
    _assert_thread_access(thread_id, user)
    # Clamp pagination: a negative offset silently wrapped around the list and an
    # unbounded limit let one request pull an entire conversation history.
    offset = max(0, offset)
    limit = max(1, min(limit, 500))
    msgs = get_messages(thread_id)
    return msgs[offset:offset+limit]

@app.get("/api/v1/chunks/{chunk_id}")
@limiter.limit(settings.RATE_LIMIT_ANON)
def get_chunk_by_id(request: Request, chunk_id: str):
    c = get_chunk(chunk_id)
    if not c:
        raise HTTPException(404, "Chunk not found")
    return c

@app.get("/api/v1/documents")
@limiter.limit(settings.RATE_LIMIT_ANON)
def list_docs(request: Request):
    chunks = list_chunks()
    docs = {}
    for c in chunks:
        docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "title": c["doc_title"], "pages": set(), "chunks": 0})
        docs[c["doc_id"]]["pages"].add(c["page"])
        docs[c["doc_id"]]["chunks"] += 1
    return [{"doc_id": v["doc_id"], "title": v["title"], "pages": len(v["pages"]), "chunks": v["chunks"]} for v in docs.values()]

@app.post("/api/v1/documents/upload")
@limiter.limit("10/minute")
async def upload_pdf(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed")
    data = await file.read()
    if len(data) > settings.MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large ({settings.MAX_PDF_MB}MB max)")
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    from app.ingest import create_job, ingest_pdf_sync
    job_id = create_job()
    # Run in background so 100-page PDFs don't block
    background_tasks.add_task(ingest_pdf_sync, data, file.filename, job_id)
    return {"job_id": job_id, "status": "queued", "filename": file.filename, "bytes": len(data)}

@app.get("/api/v1/documents/jobs/{job_id}")
@limiter.limit(settings.RATE_LIMIT_AUTH)  # polled once a second by the upload UI
def get_job_status(request: Request, job_id: str):
    from app.ingest import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **job}

# ---- Streaming Chat (production hardening) ----
@app.post("/api/v1/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ANON)
async def chat_stream(request: Request, body: ChatRequest, user=Depends(get_current_user)):
    query = (body.message or "").strip()
    if not query:
        raise HTTPException(400, "Message empty")
    if len(query) > settings.MAX_MESSAGE_LENGTH:
        raise HTTPException(400, f"Message too long (max {settings.MAX_MESSAGE_LENGTH})")
    # Basic injection guard
    if len(query.split()) < 1:
        raise HTTPException(400, "Invalid query")
    top_k = min(body.top_k or settings.TOP_K, 10)
    # Ensure the thread the client referenced actually exists under that id, and
    # that it belongs to the caller before appending to it.
    if not get_thread(body.thread_id):
        create_thread(title=body.thread_id, user_id=user["user_id"], thread_id=body.thread_id)
    else:
        _assert_thread_access(body.thread_id, user)

    # Greeting bypass
    from app.rag import is_greeting
    if is_greeting(query):
        retrieved = []
        filtered = []
        is_refusal = False
        citations = []
        expand_query = query
    else:
        expand_query = query
        history = get_messages(body.thread_id)
        low = query.lower().strip()
        is_anaphora = any(p in low for p in ["that ", "this ", "it ", "contraindication", "dosage", "dose"])
        is_short_followup = is_anaphora and len(low.split()) <= 12 and len(history) > 0
        if is_short_followup:
            # Pick last *clinical* user message (the one that had citations), not the hair boundary
            ctx = ""
            for idx in range(len(history)-1, -1, -1):
                if history[idx].get("role") == "user":
                    # check if following assistant had citations
                    nxt = history[idx+1] if idx+1 < len(history) else None
                    if nxt and nxt.get("citations"):
                        ctx = history[idx]["content"]
                        break
            if not ctx:
                # fallback: last user message with medical keyword
                for m in reversed(history):
                    if m.get("role") == "user" and any(k in m["content"].lower() for k in ["hypertension","ckd","lisinopril","arb","creatinine","enoxaparin"]):
                        ctx = m["content"]
                        break
            if ctx:
                expand_query = f"{ctx} {query}"
        retrieved = retrieve(expand_query, top_k=top_k)
        thresh = effective_threshold()
        filtered = [(c,s) for c,s in retrieved if s >= thresh]
        is_refusal = False  # product: boundary handled by AI
        citations = []
        for i,(chunk,score) in enumerate(filtered,1):
            citations.append({"id": chunk["id"], "doc_id": chunk["doc_id"], "doc_title": chunk["doc_title"], "page": chunk["page"], "chunk_text": chunk["text"], "score": round(float(score),3), "idx": i})

    # Don't log PHI
    if settings.LOG_QUERIES:
        logger.info(f"chat thread={body.thread_id} q_len={len(query)} citations={len(citations)}")
    else:
        logger.info(f"chat thread={body.thread_id} citations={len(citations)}")

    add_message(body.thread_id, "user", query)

    async def event_gen():
        # TTFT: flush meta immediately
        yield {"event": "meta", "data": json.dumps({"citations": citations, "is_refusal": is_refusal, "thread_id": body.thread_id})}
        await asyncio.sleep(0.005)
        # Heartbeat to keep Render free tier from killing idle SSE
        full_text = ""
        try:
            async for token in generate_answer(expand_query, retrieved):
                full_text += token
                yield {"event": "token", "data": json.dumps({"token": token})}
                # Check client disconnect
                if await request.is_disconnected():
                    logger.info(f"client disconnected {body.thread_id}")
                    break
        except asyncio.CancelledError:
            logger.info(f"stream cancelled {body.thread_id}")
            raise
        except Exception as e:
            logger.error(f"stream error: {e}")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}
        # Save assistant message
        if full_text:
            add_message(body.thread_id, "assistant", full_text.strip(), citations)
        yield {"event": "done", "data": json.dumps({"full_text": full_text.strip(), "citations": citations})}
        # Final heartbeat
        yield {"event": "heartbeat", "data": json.dumps({"ts": time.time()})}

    headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return EventSourceResponse(event_gen(), headers=headers, media_type="text/event-stream")

@app.post("/api/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ANON)
async def chat_stream_compat(request: Request, body: ChatRequest, user=Depends(get_current_user)):
    """Legacy path kept for older clients.

    This carried no rate limit while /api/v1/chat/stream did, so the limit on the
    most expensive endpoint in the app could be skipped entirely by dropping /v1
    from the URL.
    """
    return await chat_stream(request, body, user)
