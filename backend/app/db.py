"""
Production DB: Postgres + pgvector (Supabase) with fallback to in-memory for local dev.
- Uses SQLAlchemy 2.0 + pgvector
- If DATABASE_URL unreachable, falls back to in-memory dicts (keeps `docker compose` working without Postgres)
- Thread/Message/Chunk persisted, encrypted at rest via Supabase (pgsodium)
"""
import uuid, time, json, logging
from typing import List, Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models import Base, Thread, Message, Chunk

logger = logging.getLogger("aura.db")


def _normalize_db_url(url: str) -> str:
    """Pin the driver to psycopg3, the only Postgres driver we actually ship.

    A bare ``postgresql://`` URL makes SQLAlchemy resolve the *psycopg2* dialect,
    which is not in requirements.txt. Locally that silently works (psycopg2 gets
    pulled into the dev venv), but on a clean install it raises ModuleNotFoundError,
    which _init_engine swallows -- so the whole app quietly runs on in-memory dicts
    with no database at all. Normalizing here keeps dev and prod on one driver.
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):  # Heroku/Supabase-style legacy scheme
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url

# In-memory fallback (preserves demo without DB)
_threads: Dict[str, dict] = {}
_messages: Dict[str, List[dict]] = {}
_chunks: Dict[str, dict] = {}

SEED_CHUNKS = [
    {
        "id": "chk_001",
        "doc_id": "doc_fda_2024",
        "doc_title": "FDA Hypertension Guideline 2024",
        "page": 12,
        "text": "For adults with hypertension and chronic kidney disease, first-line therapy includes ACE inhibitors (e.g., lisinopril 10mg daily) or ARBs. Monitor serum creatinine and potassium within 2-4 weeks of initiation.",
    },
    {
        "id": "chk_002",
        "doc_id": "doc_fda_2024",
        "doc_title": "FDA Hypertension Guideline 2024",
        "page": 14,
        "text": "Contraindications for ACE inhibitors include history of angioedema, bilateral renal artery stenosis, and pregnancy. Concomitant use with aliskiren is contraindicated in patients with diabetes.",
    },
    {
        "id": "chk_003",
        "doc_id": "doc_trial_nct042",
        "doc_title": "NCT042 Trial Results — Anticoagulation Protocol",
        "page": 42,
        "text": "Enoxaparin 40mg subcutaneously once daily is recommended for VTE prophylaxis in hospitalized COVID-19 patients with D-dimer >3x ULN, unless bleeding risk is high. Efficacy endpoint measured at day 28.",
    },
]

for c in SEED_CHUNKS:
    _chunks[c["id"]] = c

# Try to init Postgres engine
_engine = None
_SessionLocal = None
_db_available = False
_db_error: Optional[str] = None  # surfaced by /health so a failed DB is never silent

def _init_engine():
    global _engine, _SessionLocal, _db_available, _db_error
    try:
        _engine = create_engine(
            _normalize_db_url(settings.DATABASE_URL),
            pool_size=10, max_overflow=20, pool_pre_ping=True, pool_timeout=10,
            connect_args={
                "connect_timeout": 5,
                # psycopg3 prepares statements after a few executions, which breaks
                # behind a transaction-mode pooler (Supabase port 6543, pgbouncer):
                # the server-side statement vanishes between transactions and the
                # next execution fails with "prepared statement already exists".
                # Disabling it costs a little planning time and makes any pooler
                # configuration work.
                "prepare_threshold": None,
            },
        )
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # try pgvector extension
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:
                # Managed Postgres often refuses CREATE EXTENSION to non-superusers.
                # Worth continuing -- the extension may already be enabled -- but not
                # worth hiding: without pgvector the Vector column cannot be created,
                # create_all below fails, and the whole app drops to in-memory.
                logger.warning(
                    "could not ensure pgvector extension (%s); enable it manually if "
                    "table creation fails", e,
                )
        # create tables if not exist
        Base.metadata.create_all(bind=_engine)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        _db_available = True
        _db_error = None
        # seed if empty
        try:
            with _SessionLocal() as s:
                cnt = s.query(Chunk).count()
                if cnt == 0:
                    for c in SEED_CHUNKS:
                        s.add(Chunk(id=c["id"], doc_id=c["doc_id"], doc_title=c["doc_title"], page=c["page"], text=c["text"]))
                    s.commit()
        except Exception as e:
            logger.warning("seed skipped: %s", e)
    except Exception as e:
        # This fallback means NOTHING is persisted: threads, messages and uploaded
        # documents live in process memory, are shared by every caller, and vanish
        # on restart. That is fine for a local demo and wrong everywhere else, so
        # say so loudly rather than printing one line and carrying on.
        _db_error = f"{type(e).__name__}: {e}"
        _db_available = False
        logger.error(
            "DATABASE UNAVAILABLE -- falling back to EPHEMERAL IN-MEMORY storage. "
            "Nothing will be persisted and all users share one dataset. Cause: %s",
            _db_error,
        )

_init_engine()

def _db_op_failed(op: str, exc: Exception) -> None:
    """Report a failed DB operation on a database we believe is up.

    Every function below catches broadly and then writes to the in-memory dicts.
    That is correct when Postgres was never reachable, but when the engine IS live
    a failure means the write silently went to RAM instead of the database while
    the caller was told it succeeded -- so some rows are in Postgres and some are
    not, with nothing recorded anywhere. Log those loudly; they are bugs, not
    fallback.
    """
    logger.error("db operation %s failed against a live engine, value went to memory only: %s", op, exc)


def is_db_available() -> bool:
    return _db_available

def db_error() -> Optional[str]:
    """Why the DB is unavailable, or None when it is healthy."""
    return _db_error

def storage_mode() -> str:
    return "postgres" if _db_available else "in-memory (ephemeral)"

def chunk_embedding_stats() -> dict:
    """How much of the corpus is actually reachable by vector search.

    The pgvector query filters `WHERE embedding IS NOT NULL`, so any chunk stored
    without an embedding -- including the seeded clinical guidelines, which are
    inserted with no embedding at all -- becomes invisible to retrieval the moment
    a single other chunk has one. A mixed corpus therefore silently answers from a
    subset, which is why this is reported rather than left to be discovered.
    """
    if not (_db_available and _SessionLocal):
        return {"total": len(_chunks), "embedded": 0, "unretrievable_in_vector_mode": len(_chunks)}
    try:
        with _SessionLocal() as s:
            total = s.query(Chunk).count()
            embedded = s.query(Chunk).filter(Chunk.embedding.isnot(None)).count()
        return {"total": total, "embedded": embedded,
                "unretrievable_in_vector_mode": total - embedded}
    except Exception as e:
        _db_op_failed("chunk_embedding_stats", e)
        return {"total": 0, "embedded": 0, "unretrievable_in_vector_mode": 0}

_last_reconnect_attempt = 0.0
_RECONNECT_INTERVAL = 30.0


def _maybe_reconnect() -> None:
    """Retry engine init if the database was down at import.

    _db_available was latched once at module import and never re-evaluated, so a
    database that was briefly unreachable during boot -- entirely normal when the
    API and Postgres start together, or on a Render free-tier cold start -- left
    the process permanently in ephemeral in-memory mode until someone restarted it.
    """
    global _last_reconnect_attempt
    if _db_available:
        return
    now = time.time()
    if now - _last_reconnect_attempt < _RECONNECT_INTERVAL:
        return
    _last_reconnect_attempt = now
    logger.info("retrying database connection")
    _init_engine()
    if _db_available:
        logger.info("database recovered; leaving in-memory fallback mode")


def try_pg_connection() -> bool:
    _maybe_reconnect()
    if not _db_available or not _engine:
        return False
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        # Returning False is the intended signal here, but record why so a probe
        # flapping between ok and degraded is diagnosable.
        logger.warning("pg liveness check failed: %s", e)
        return False

# ---- Thread ops ----
def create_thread(title: str = "New consultation", user_id: str = "anonymous",
                  thread_id: Optional[str] = None) -> dict:
    """Create a thread, optionally with a caller-supplied id.

    Callers that need a specific id (the chat endpoint materialising the thread a
    client already referenced) must pass thread_id. Passing it as `title` created a
    thread under a fresh random id instead, leaving the referenced id absent and
    add_message to silently create a second, untitled, anonymous-owned thread.
    """
    tid = thread_id or f"thr_{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                s.add(Thread(id=tid, title=title, user_id=user_id))
                s.commit()
                return {"id": tid, "title": title, "created_at": now, "user_id": user_id}
        except Exception as e:
            _db_op_failed("create_thread", e)
    t = {"id": tid, "title": title, "created_at": now, "user_id": user_id}
    _threads[tid] = t
    _messages[tid] = []
    return t

def list_threads(user_id: Optional[str] = None):
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                q = s.query(Thread)
                if user_id:
                    q = q.filter(Thread.user_id == user_id)
                q = q.order_by(Thread.created_at.desc()).limit(100)
                return [{"id": t.id, "title": t.title, "created_at": t.created_at.isoformat() + "Z" if t.created_at else "", "user_id": t.user_id} for t in q.all()]
        except Exception as e:
            _db_op_failed("list_threads", e)
    # fallback
    vals = list(_threads.values())
    if user_id:
        vals = [v for v in vals if v.get("user_id") == user_id]
    return vals

def get_thread(tid: str):
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                t = s.query(Thread).filter(Thread.id == tid).first()
                if t:
                    return {"id": t.id, "title": t.title, "created_at": t.created_at.isoformat() + "Z" if t.created_at else "", "user_id": t.user_id}
        except Exception as e:
            _db_op_failed("get_thread", e)
    return _threads.get(tid)

def add_message(tid: str, role: str, content: str, citations=None, user_id: str = "anonymous"):
    citations_json = json.dumps(citations or [])
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                # ensure thread exists
                t = s.query(Thread).filter(Thread.id == tid).first()
                if not t:
                    # Own it by the caller. Hardcoding "anonymous" here detached
                    # every chat-created thread from the user who started it, so it
                    # never appeared in their own thread list and ownership checks
                    # could not scope it.
                    s.add(Thread(id=tid, title=tid, user_id=user_id))
                    s.commit()
                s.add(Message(id=f"msg_{uuid.uuid4().hex[:8]}", thread_id=tid, role=role, content=content, citations=citations_json))
                s.commit()
                return
        except Exception as e:
            _db_op_failed("add_message", e)
    if tid not in _messages:
        _messages[tid] = []
    _messages[tid].append({"role": role, "content": content, "citations": citations or [], "ts": time.time()})

def get_messages(tid: str):
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                msgs = s.query(Message).filter(Message.thread_id == tid).order_by(Message.created_at.asc()).all()
                return [{"role": m.role, "content": m.content, "citations": json.loads(m.citations) if m.citations else [], "ts": m.created_at.timestamp() if m.created_at else 0} for m in msgs]
        except Exception as e:
            _db_op_failed("get_messages", e)
    return _messages.get(tid, [])

def list_chunks():
    """Return the retrieval corpus.

    Raises when the database is up but unreadable, rather than quietly returning
    the three hardcoded SEED_CHUNKS. Silently substituting a demo corpus meant a
    transient DB error made the assistant answer clinical questions from the wrong
    documents while still presenting them as cited sources -- a wrong answer is
    worse here than no answer.
    """
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                rows = s.query(Chunk).all()
                return [{"id": r.id, "doc_id": r.doc_id, "doc_title": r.doc_title, "page": r.page, "text": r.text} for r in rows]
        except Exception as e:
            _db_op_failed("list_chunks", e)
            raise RuntimeError("retrieval corpus is unavailable") from e
    return list(_chunks.values())

def get_chunk(cid: str):
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                r = s.query(Chunk).filter(Chunk.id == cid).first()
                if r:
                    return {"id": r.id, "doc_id": r.doc_id, "doc_title": r.doc_title, "page": r.page, "text": r.text}
        except Exception as e:
            _db_op_failed("get_chunk", e)
    return _chunks.get(cid)

def upsert_chunks(docs: List[dict]) -> bool:
    """Persist chunks. Returns True if they reached the database."""
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                for d in docs:
                    # embedding handled separately
                    existing = s.query(Chunk).filter(Chunk.id == d["id"]).first()
                    if existing:
                        existing.text = d["text"]
                        existing.page = d["page"]
                        existing.doc_title = d["doc_title"]
                    else:
                        s.add(Chunk(id=d["id"], doc_id=d["doc_id"], doc_title=d["doc_title"], page=d["page"], text=d["text"]))
                s.commit()
                return True
        except Exception as e:
            _db_op_failed("upsert_chunks", e)
            return False
    for d in docs:
        _chunks[d["id"]] = d
    return True

def upsert_chunk_with_embedding(chunk: dict, embedding: List[float]) -> bool:
    """Insert a chunk with its vector. Returns True if it reached the database.

    The in-memory fallback below is unreachable by retrieval whenever the DB is up,
    because list_chunks() returns DB rows and never consults _chunks in that state.
    A failure here therefore means the chunk is lost, so the caller is told rather
    than left to report the ingest as successful.
    """
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                s.merge(Chunk(id=chunk["id"], doc_id=chunk["doc_id"], doc_title=chunk["doc_title"], page=chunk["page"], text=chunk["text"], embedding=embedding))
                s.commit()
                return True
        except Exception as e:
            _db_op_failed("upsert_chunk_with_embedding", e)
            return False
    _chunks[chunk["id"]] = chunk
    return True
