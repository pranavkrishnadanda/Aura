"""Integration tests for app/ingest.py, using real PDFs built in-memory with
PyMuPDF (fitz) rather than fixture files.
"""
import fitz
import pytest

from app import db, ingest
from app.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _no_network_embeddings(monkeypatch):
    """This environment has a real GEMINI_API_KEY configured. Force it empty so
    _embed_texts() takes its network-free `[None] * len(texts)` path everywhere in
    this file by default; individual tests that need embedded chunks patch
    ingest._embed_texts directly instead of relying on the real API.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")


@pytest.fixture(autouse=True)
def _isolate_chunks():
    snapshot = dict(db._chunks)
    yield
    db._chunks.clear()
    db._chunks.update(snapshot)


def _make_pdf(pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text)
    data = doc.tobytes()
    doc.close()
    return data


def _all_chunks_by_id():
    """Read the corpus through the public accessor.

    Reading db._chunks directly only sees writes in the no-database configuration;
    when a real database is attached the rows land in Postgres and the dict stays
    empty, so these assertions silently measured nothing.
    """
    return {c["id"]: c for c in db.list_chunks()}


def _new_chunks_since(before_ids):
    return [v for k, v in _all_chunks_by_id().items() if k not in before_ids]


def test_single_pdf_yields_one_distinct_doc_id():
    """Regression: doc_id used to be minted per chunk (inside the per-chunk loop),
    so a single multi-chunk upload showed up as N separate documents under
    GET /api/v1/documents, which groups chunks by doc_id.
    """
    pdf = _make_pdf([
        "Clinical note page one with enough distinct content to form its own chunk here.",
        "Clinical note page two with different distinct content to form another chunk here.",
    ])
    before_ids = set(_all_chunks_by_id())
    result = ingest.ingest_pdf_sync(pdf, "multi.pdf")

    new_chunks = _new_chunks_since(before_ids)
    assert len(new_chunks) == result["chunks"]
    assert result["chunks"] >= 2
    doc_ids = {c["doc_id"] for c in new_chunks}
    assert doc_ids == {result["doc_id"]}


def test_page_numbers_are_one_based_and_correct():
    pdf = _make_pdf([
        "Page one content long enough to clear chunk_text's fifty character minimum easily.",
        "Page two content long enough to clear chunk_text's fifty character minimum easily.",
        "Page three content long enough to clear chunk_text's fifty character minimum too.",
    ])
    before_ids = set(_all_chunks_by_id())
    ingest.ingest_pdf_sync(pdf, "pages.pdf")

    new_chunks = _new_chunks_since(before_ids)
    pages = sorted(c["page"] for c in new_chunks)
    assert pages == [1, 2, 3]


def test_corrupt_pdf_marks_job_failed():
    job_id = ingest.create_job()
    with pytest.raises(Exception):
        ingest.ingest_pdf_sync(b"this is not a pdf file at all", "bad.pdf", job_id)

    job = ingest.get_job(job_id)
    assert job["status"] == "failed"
    assert job.get("error")


def test_all_writes_failing_marks_job_failed_not_completed(monkeypatch):
    """Regression: the job used to be marked "completed" as soon as chunking
    finished, regardless of whether any chunk actually reached storage -- telling
    the user their document was indexed when retrieval could never see it.
    """
    monkeypatch.setattr(ingest, "upsert_chunks", lambda docs: False)
    monkeypatch.setattr(ingest, "upsert_chunk_with_embedding", lambda chunk, vec: False)

    pdf = _make_pdf(["Enough content here to survive the fifty character chunk filter easily."])
    job_id = ingest.create_job()
    result = ingest.ingest_pdf_sync(pdf, "fail.pdf", job_id)
    assert result["chunks"] >= 1

    job = ingest.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "no chunks could be stored"


def test_partial_write_failure_marks_job_partial(monkeypatch):
    """Some chunks reach storage and some don't -> "partial", distinct from both
    "completed" and "failed" so the caller knows the document is only partly
    searchable.
    """
    pdf = _make_pdf([
        "First page unique content long enough to pass the fifty character filter here.",
        "Second page unique content long enough to pass the fifty character filter too.",
    ])

    def fake_embed(texts):
        # First chunk "embeds" successfully, the rest fall back to the plain path.
        return [[0.1, 0.2]] + [None] * (len(texts) - 1)

    monkeypatch.setattr(ingest, "_embed_texts", fake_embed)
    monkeypatch.setattr(ingest, "upsert_chunk_with_embedding", lambda chunk, vec: True)
    monkeypatch.setattr(ingest, "upsert_chunks", lambda docs: False)

    job_id = ingest.create_job()
    result = ingest.ingest_pdf_sync(pdf, "partial.pdf", job_id)
    assert result["chunks"] == 2  # sanity: exactly one chunk per page as expected

    job = ingest.get_job(job_id)
    assert job["status"] == "partial"
    assert "1 of 2" in job["error"]


def test_empty_and_whitespace_only_pages_are_skipped():
    pdf = _make_pdf([
        "Real content page with enough characters to pass the fifty char minimum easily.",
        "",  # truly blank page
        "   ",  # whitespace-only page
    ])
    result = ingest.ingest_pdf_sync(pdf, "sparse.pdf")
    assert result["pages"] == 3
    assert result["chunks"] == 1  # only the first page contributed a chunk


def test_pdf_handle_is_closed_after_ingest(monkeypatch):
    """Regression: guards the try/finally around fitz.Document.close(). Without it,
    every ingest call leaks the open PDF buffer/handle.
    """
    real_open = fitz.open
    captured = {}

    def spy_open(*args, **kwargs):
        d = real_open(*args, **kwargs)
        captured["doc"] = d
        return d

    monkeypatch.setattr(ingest.fitz, "open", spy_open)

    pdf = _make_pdf(["Some real content long enough to pass the fifty character chunk filter here."])
    ingest.ingest_pdf_sync(pdf, "close.pdf")

    assert "doc" in captured
    assert captured["doc"].is_closed is True


def test_jobs_dict_is_capped_at_max_jobs():
    """Regression guard: _jobs is an unbounded dict unless create_job() evicts old
    entries once _MAX_JOBS is reached, which would leak memory on a long-running
    instance.
    """
    for _ in range(ingest._MAX_JOBS + 50):
        ingest.create_job()
    assert len(ingest._jobs) <= ingest._MAX_JOBS
