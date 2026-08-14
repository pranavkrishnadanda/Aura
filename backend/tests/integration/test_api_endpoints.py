"""Integration tests for the HTTP surface in app/main.py.

Runs against the real FastAPI app with the in-memory fallback store (there is no
live Postgres in this environment, so app.db already operates in
"in-memory (ephemeral)" mode -- see app/db.py's _init_engine()). DB failure is
simulated where needed via monkeypatch on the names app.main imported.
"""
import io

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_network_embeddings(monkeypatch):
    """This environment has a real GEMINI_API_KEY configured. Any test here that
    triggers document ingestion (POST /api/v1/documents/upload) would otherwise
    make a real outbound embedding call. Force the key empty so ingest always
    takes the deterministic, network-free TF-IDF/plain-storage path.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")


@pytest.fixture(autouse=True)
def _isolate_chunks():
    """list_chunks()/GET /api/v1/documents read the shared app.db._chunks dict,
    which tests/conftest.py does NOT reset between tests (only _threads/_messages/
    _jobs are cleared there). Snapshot and restore it here so a PDF uploaded by one
    test can't change the document count seen by another.
    """
    from app.db import _chunks

    snapshot = dict(_chunks)
    yield
    _chunks.clear()
    _chunks.update(snapshot)


def _make_pdf(pages_text):
    """Build a real, minimal PDF in memory with PyMuPDF."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text)
    data = doc.tobytes()
    doc.close()
    return data


# ---- /health ----

def test_health_always_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_reports_degraded_storage_mode_and_db_error_when_pg_unreachable(monkeypatch):
    """When Postgres can't be reached, /health must say so explicitly rather than
    reporting "ok" over an ephemeral in-memory store."""
    monkeypatch.setattr("app.main.try_pg_connection", lambda: False)
    monkeypatch.setattr("app.main.storage_mode", lambda: "in-memory (ephemeral)")
    monkeypatch.setattr("app.main.db_error", lambda: "OperationalError: simulated outage")

    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["storage_mode"] == "in-memory (ephemeral)"
    assert body["db_error"] == "OperationalError: simulated outage"
    assert body["pg_reachable"] is False


def test_health_ok_when_pg_reachable(monkeypatch):
    monkeypatch.setattr("app.main.try_pg_connection", lambda: True)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pg_reachable"] is True


def test_health_reports_retrieval_mode_and_matching_threshold():
    """threshold must be the floor actually enforced for the reported retrieval_mode,
    not a fixed number that disagrees with configured_threshold/tfidf_threshold."""
    body = client.get("/health").json()
    assert body["retrieval_mode"] in ("pgvector", "tfidf", "token-overlap")
    assert body["configured_threshold"] == settings.RETRIEVAL_THRESHOLD
    assert body["tfidf_threshold"] == settings.TFIDF_THRESHOLD
    if body["retrieval_mode"] == "pgvector":
        assert body["threshold"] == settings.RETRIEVAL_THRESHOLD
    else:
        assert body["threshold"] == settings.TFIDF_THRESHOLD


def test_health_uptime_is_a_small_elapsed_duration_not_an_epoch_timestamp():
    """Regression: uptime_seconds must be time since process start, not time.time()
    itself -- an accidental swap makes this field look like ~1.7 billion seconds."""
    body = client.get("/health").json()
    assert isinstance(body["uptime_seconds"], (int, float))
    assert 0 <= body["uptime_seconds"] < 3600  # this test process is seconds old


# ---- /ready ----

def test_ready_returns_503_when_pg_down(monkeypatch):
    """Regression: /ready previously returned a hardcoded ready=true, so the
    readiness probe could never actually fail."""
    monkeypatch.setattr("app.main.try_pg_connection", lambda: False)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert body["pg"] is False


def test_ready_returns_200_when_pg_and_llm_up(monkeypatch):
    monkeypatch.setattr("app.main.try_pg_connection", lambda: True)
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["pg"] is True


# ---- POST /api/v1/threads ----

def test_create_thread_with_default_title():
    r = client.post("/api/v1/threads", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "New consultation"
    assert body["id"]


def test_create_thread_with_custom_title():
    r = client.post("/api/v1/threads", json={"title": "Follow-up visit"})
    assert r.status_code == 200
    assert r.json()["title"] == "Follow-up visit"


def test_create_thread_title_over_200_chars_is_rejected():
    """An over-length title is rejected by schema validation, as 422.

    post_thread previously also carried `if len(title) > 200: raise 400`, which
    was unreachable because ThreadCreate.title's max_length rejects first -- dead
    code advertising a status the API never returned. One validation layer owns
    this now, and 422 is the answer.
    """
    r = client.post("/api/v1/threads", json={"title": "x" * 201})
    assert r.status_code == 422
    assert r.status_code != 400, "two validation layers disagreeing about the status"


def test_create_thread_title_at_the_limit_is_accepted():
    """Boundary: exactly 200 characters must still be allowed."""
    r = client.post("/api/v1/threads", json={"title": "x" * 200})
    assert r.status_code == 200
    assert r.json()["title"] == "x" * 200


# ---- GET /api/v1/threads/{id}/messages ----

def test_get_messages_for_unknown_thread_is_404():
    r = client.get("/api/v1/threads/thr_does_not_exist/messages")
    assert r.status_code == 404


def test_negative_offset_does_not_wrap_around_the_list():
    """Regression: offset=max(0, offset) must run before slicing. Without the
    clamp, msgs[-1:0] on a 3-item list silently returns an empty page instead of
    the first message.
    """
    from app.db import add_message

    tid = client.post("/api/v1/threads", json={"title": "pagination"}).json()["id"]
    for i in range(3):
        add_message(tid, "user", f"msg{i}", user_id="anonymous")

    r = client.get(f"/api/v1/threads/{tid}/messages?offset=-1&limit=1")
    assert r.status_code == 200
    msgs = r.json()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "msg0"


def test_huge_limit_is_capped_at_500():
    """Regression: an unbounded limit let one request pull an entire conversation
    history in one page."""
    from app.db import add_message

    tid = client.post("/api/v1/threads", json={"title": "big thread"}).json()["id"]
    for i in range(501):
        add_message(tid, "user", f"m{i}", user_id="anonymous")

    r = client.get(f"/api/v1/threads/{tid}/messages?limit=999999")
    assert r.status_code == 200
    assert len(r.json()) == 500


# ---- GET /api/v1/chunks/{chunk_id} ----

def test_get_seeded_chunk_returns_200():
    r = client.get("/api/v1/chunks/chk_001")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "chk_001"
    assert body["doc_id"] == "doc_fda_2024"


def test_get_unknown_chunk_returns_404():
    r = client.get("/api/v1/chunks/chk_does_not_exist")
    assert r.status_code == 404


# ---- GET /api/v1/documents ----

def test_list_documents_groups_chunks_by_doc_id():
    # conftest restores both the in-memory dict and the real tables to exactly the
    # seed set before every test, so these counts hold in either storage mode.
    r = client.get("/api/v1/documents")
    assert r.status_code == 200
    docs = {d["doc_id"]: d for d in r.json()}
    assert docs["doc_fda_2024"]["chunks"] == 2
    assert docs["doc_fda_2024"]["pages"] == 2
    assert docs["doc_trial_nct042"]["chunks"] == 1
    assert docs["doc_trial_nct042"]["pages"] == 1


def test_list_documents_returns_503_when_list_chunks_raises(monkeypatch):
    def boom():
        raise RuntimeError("corpus unavailable")

    monkeypatch.setattr("app.main.list_chunks", boom)
    r = client.get("/api/v1/documents")
    assert r.status_code == 503


# ---- POST /api/v1/documents/upload ----

def test_upload_rejects_non_pdf_extension():
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_upload_rejects_content_not_starting_with_pdf_magic_bytes():
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("fake.pdf", b"this is not really a pdf file", "application/pdf")},
    )
    assert r.status_code == 400


def test_upload_rejects_empty_file():
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert r.status_code == 400


def test_upload_rejects_file_over_max_pdf_mb(monkeypatch):
    """Regression: the body used to be fully buffered before the size check ran,
    so MAX_PDF_MB never actually bounded memory. Assert the limit is enforced with
    a small, fast-to-generate payload rather than a real 50MB+ file.
    """
    monkeypatch.setattr(settings, "MAX_PDF_MB", 1)
    oversized = b"%PDF-1.4\n" + b"a" * (2 * 1024 * 1024)
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert r.status_code == 400


def test_upload_accepts_valid_pdf():
    pdf_bytes = _make_pdf(["Clinical note with enough content to be a real page."])
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("consult.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["status"] == "queued"
    assert body["filename"] == "consult.pdf"
    assert body["bytes"] == len(pdf_bytes)


# ---- GET /api/v1/documents/jobs/{job_id} ----

def test_get_unknown_job_returns_404():
    r = client.get("/api/v1/documents/jobs/job_does_not_exist")
    assert r.status_code == 404


def test_get_job_status_for_real_upload():
    pdf_bytes = _make_pdf(["Enough page content here to survive chunk_text's 50-char filter."])
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("job_test.pdf", pdf_bytes, "application/pdf")},
    ).json()

    r = client.get(f"/api/v1/documents/jobs/{upload['job_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == upload["job_id"]
    # TestClient runs BackgroundTasks to completion before the response returns, so
    # by the time the upload POST above came back the ingest job already finished.
    assert body["status"] in ("completed", "partial", "failed")
    assert body["pages"] == 1


# ---- Validation ----

def test_empty_message_is_rejected_with_422():
    r = client.post("/api/v1/chat/stream", json={"message": "", "thread_id": "v1"})
    assert r.status_code == 422


def test_message_over_max_length_is_rejected_with_422():
    """Regression: MAX_MESSAGE_LENGTH was hardcoded to 4000 in the field bound, so
    raising the setting above 4000 had no effect -- Pydantic rejected long messages
    before the endpoint's own length check could run."""
    too_long = "a" * (settings.MAX_MESSAGE_LENGTH + 1)
    r = client.post("/api/v1/chat/stream", json={"message": too_long, "thread_id": "v1"})
    assert r.status_code == 422


def test_top_k_out_of_range_is_rejected_with_422():
    r = client.post(
        "/api/v1/chat/stream",
        json={"message": "hi", "thread_id": "v1", "top_k": 11},
    )
    assert r.status_code == 422

    r2 = client.post(
        "/api/v1/chat/stream",
        json={"message": "hi", "thread_id": "v1", "top_k": 0},
    )
    assert r2.status_code == 422
