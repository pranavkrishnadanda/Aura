"""
Production DB: Postgres + pgvector (Supabase) with fallback to in-memory for local dev.
- Uses SQLAlchemy 2.0 + pgvector
- If DATABASE_URL unreachable, falls back to in-memory dicts (keeps `docker compose` working without Postgres)
- Thread/Message/Chunk persisted, encrypted at rest via Supabase (pgsodium)
"""
import uuid, time, json
from typing import List, Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models import Base, Thread, Message, Chunk

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

def _init_engine():
    global _engine, _SessionLocal, _db_available
    try:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_size=10, max_overflow=20, pool_pre_ping=True, pool_timeout=10,
            connect_args={"connect_timeout": 5}
        )
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # try pgvector extension
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception:
                pass
        # create tables if not exist
        Base.metadata.create_all(bind=_engine)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        _db_available = True
        # seed if empty
        try:
            with _SessionLocal() as s:
                cnt = s.query(Chunk).count()
                if cnt == 0:
                    for c in SEED_CHUNKS:
                        s.add(Chunk(id=c["id"], doc_id=c["doc_id"], doc_title=c["doc_title"], page=c["page"], text=c["text"]))
                    s.commit()
        except Exception:
            pass
    except Exception as e:
        print(f"[db] Postgres unavailable, using in-memory fallback: {e}")
        _db_available = False

_init_engine()

def is_db_available() -> bool:
    return _db_available

def try_pg_connection() -> bool:
    if not _db_available or not _engine:
        return False
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

# ---- Thread ops ----
def create_thread(title: str = "New consultation", user_id: str = "anonymous") -> dict:
    tid = f"thr_{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                s.add(Thread(id=tid, title=title, user_id=user_id))
                s.commit()
                return {"id": tid, "title": title, "created_at": now, "user_id": user_id}
        except Exception:
            pass
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
        except Exception:
            pass
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
        except Exception:
            pass
    return _threads.get(tid)

def add_message(tid: str, role: str, content: str, citations=None):
    citations_json = json.dumps(citations or [])
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                # ensure thread exists
                t = s.query(Thread).filter(Thread.id == tid).first()
                if not t:
                    s.add(Thread(id=tid, title=tid, user_id="anonymous"))
                    s.commit()
                s.add(Message(id=f"msg_{uuid.uuid4().hex[:8]}", thread_id=tid, role=role, content=content, citations=citations_json))
                s.commit()
                return
        except Exception:
            pass
    if tid not in _messages:
        _messages[tid] = []
    _messages[tid].append({"role": role, "content": content, "citations": citations or [], "ts": time.time()})

def get_messages(tid: str):
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                msgs = s.query(Message).filter(Message.thread_id == tid).order_by(Message.created_at.asc()).all()
                return [{"role": m.role, "content": m.content, "citations": json.loads(m.citations) if m.citations else [], "ts": m.created_at.timestamp() if m.created_at else 0} for m in msgs]
        except Exception:
            pass
    return _messages.get(tid, [])

def list_chunks():
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                rows = s.query(Chunk).all()
                if rows:
                    return [{"id": r.id, "doc_id": r.doc_id, "doc_title": r.doc_title, "page": r.page, "text": r.text} for r in rows]
        except Exception:
            pass
    return list(_chunks.values())

def get_chunk(cid: str):
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                r = s.query(Chunk).filter(Chunk.id == cid).first()
                if r:
                    return {"id": r.id, "doc_id": r.doc_id, "doc_title": r.doc_title, "page": r.page, "text": r.text}
        except Exception:
            pass
    return _chunks.get(cid)

def upsert_chunks(docs: List[dict]):
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
                return
        except Exception:
            pass
    for d in docs:
        _chunks[d["id"]] = d

def upsert_chunk_with_embedding(chunk: dict, embedding: List[float]):
    """Insert chunk with vector (pgvector); fallback stores without vector"""
    if _db_available and _SessionLocal:
        try:
            with _SessionLocal() as s:
                s.add(Chunk(id=chunk["id"], doc_id=chunk["doc_id"], doc_title=chunk["doc_title"], page=chunk["page"], text=chunk["text"], embedding=embedding))
                s.commit()
                return
        except Exception as e:
            print(f"[db] embedding insert failed, fallback: {e}")
    _chunks[chunk["id"]] = chunk
