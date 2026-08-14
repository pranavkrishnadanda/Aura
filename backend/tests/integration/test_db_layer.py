"""Integration tests for app/db.py's public operations.

There is no live Postgres in this environment, so app.db already runs in
in-memory fallback mode (_db_available is False). Most tests exercise that real
fallback path directly. Where the regression being guarded only lives on the
"DB is up but the operation itself fails" branch, a small fake SQLAlchemy
session is substituted via monkeypatch to force that branch without needing a
real database.
"""
import pytest

from app import db
from app.models import Thread

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_chunks():
    """app.db._chunks is a module-level dict that tests/conftest.py does not reset
    (only _threads/_messages/_jobs are). Snapshot/restore it so tests that upsert
    chunks don't change what later tests (in this file or others) see.
    """
    snapshot = dict(db._chunks)
    yield
    db._chunks.clear()
    db._chunks.update(snapshot)


class _FakeQuery:
    """Minimal stand-in for a SQLAlchemy Query, enough for db.py's call patterns."""

    def __init__(self, result=None):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result

    def all(self):
        return self._result or []

    def count(self):
        return len(self._result or [])


class _FakeSession:
    """Stand-in for a live SQLAlchemy session used as `with _SessionLocal() as s:`.

    Pass raise_error to simulate a database that is reachable (engine up) but
    whose query/write fails -- the scenario db.py's functions must turn into a
    loud failure (False / an exception), not a silent fallback to SEED_CHUNKS.
    """

    def __init__(self, added=None, raise_error: Exception | None = None):
        self.added = added if added is not None else []
        self._raise_error = raise_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _maybe_raise(self):
        if self._raise_error is not None:
            raise self._raise_error

    def query(self, model):
        self._maybe_raise()
        return _FakeQuery(None)

    def add(self, obj):
        self._maybe_raise()
        self.added.append(obj)

    def merge(self, obj):
        self._maybe_raise()
        self.added.append(obj)
        return obj

    def commit(self):
        pass


def test_create_thread_with_explicit_id_uses_that_id():
    """Regression: create_thread's thread_id used to be passed positionally as
    `title`, so a caller-specified id created a fresh, randomly-id'd thread instead
    -- the id the client actually referenced was never created, and a second call
    (e.g. add_message's auto-create) would materialise an orphan under it.
    """
    t = db.create_thread(title="Chat thread", user_id="u1", thread_id="thr_fixed01")
    assert t["id"] == "thr_fixed01"
    assert t["title"] == "Chat thread"

    fetched = db.get_thread("thr_fixed01")
    assert fetched is not None
    assert fetched["id"] == "thr_fixed01"
    assert fetched["title"] == "Chat thread"


def test_add_message_autocreated_thread_uses_caller_user_id(monkeypatch):
    """Regression: add_message's auto-create-thread path (when a chat message
    arrives for a thread id the DB doesn't know yet) used to hardcode
    user_id="anonymous", so the thread it created was invisible to the actual
    owner's own thread list. This exercises the DB-available branch, where that
    creation happens, via a fake session.
    """
    added: list = []
    monkeypatch.setattr(db, "_db_available", True)
    monkeypatch.setattr(db, "_SessionLocal", lambda: _FakeSession(added))

    db.add_message("thr_new_from_chat", "user", "hello", user_id="user_77")

    created_threads = [o for o in added if isinstance(o, Thread)]
    assert len(created_threads) == 1
    assert created_threads[0].user_id == "user_77"


def test_list_threads_scopes_by_user_id():
    db.create_thread(title="Alice's thread", user_id="alice", thread_id="thr_alice")
    db.create_thread(title="Bob's thread", user_id="bob", thread_id="thr_bob")

    alice_ids = [t["id"] for t in db.list_threads(user_id="alice")]
    assert "thr_alice" in alice_ids
    assert "thr_bob" not in alice_ids


def test_list_chunks_raises_when_db_up_but_query_fails(monkeypatch):
    """Regression: list_chunks() used to catch every failure and fall through to
    the in-memory SEED_CHUNKS, so a transient DB error made the assistant answer
    clinical questions from the wrong (demo) corpus while still citing it as a
    real source. It must raise instead of substituting a different corpus.
    """
    monkeypatch.setattr(db, "_db_available", True)
    monkeypatch.setattr(db, "_SessionLocal", lambda: _FakeSession(raise_error=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="unavailable"):
        db.list_chunks()


def test_upsert_chunks_true_on_success_false_on_db_failure(monkeypatch):
    ok = db.upsert_chunks([
        {"id": "chk_test_a", "doc_id": "doc_x", "doc_title": "X", "page": 1, "text": "hello world"}
    ])
    assert ok is True
    assert db._chunks["chk_test_a"]["text"] == "hello world"

    monkeypatch.setattr(db, "_db_available", True)
    monkeypatch.setattr(db, "_SessionLocal", lambda: _FakeSession(raise_error=RuntimeError("boom")))
    ok2 = db.upsert_chunks([
        {"id": "chk_test_b", "doc_id": "doc_x", "doc_title": "X", "page": 1, "text": "hi"}
    ])
    assert ok2 is False


def test_upsert_chunk_with_embedding_true_on_success_false_on_db_failure(monkeypatch):
    ok = db.upsert_chunk_with_embedding(
        {"id": "chk_test_c", "doc_id": "doc_x", "doc_title": "X", "page": 1, "text": "hi"},
        [0.1, 0.2],
    )
    assert ok is True
    assert "chk_test_c" in db._chunks

    monkeypatch.setattr(db, "_db_available", True)
    monkeypatch.setattr(db, "_SessionLocal", lambda: _FakeSession(raise_error=RuntimeError("boom")))
    ok2 = db.upsert_chunk_with_embedding(
        {"id": "chk_test_d", "doc_id": "doc_x", "doc_title": "X", "page": 1, "text": "hi"},
        [0.1, 0.2],
    )
    assert ok2 is False


def test_chunk_embedding_stats_reports_total_embedded_unretrievable():
    # Pin the corpus to exactly the known seed set. Other test modules outside
    # this file's ownership may leave extra chunks in the shared app.db._chunks
    # dict (it isn't reset by tests/conftest.py); the autouse _isolate_chunks
    # fixture above restores whatever was there once this test finishes, so this
    # reset is scoped to this test only.
    db._chunks.clear()
    db._chunks.update({c["id"]: c for c in db.SEED_CHUNKS})

    stats = db.chunk_embedding_stats()
    assert stats["total"] == 3  # the three SEED_CHUNKS, none embedded
    assert stats["embedded"] == 0
    assert stats["unretrievable_in_vector_mode"] == 3


def test_get_thread_and_get_messages_round_trip():
    created = db.create_thread(title="Round trip", user_id="u9", thread_id="thr_rt")
    fetched = db.get_thread("thr_rt")
    assert fetched["id"] == created["id"]
    assert fetched["title"] == "Round trip"

    db.add_message("thr_rt", "user", "hello there", user_id="u9")
    db.add_message("thr_rt", "assistant", "hi back", [{"id": "chk_001"}], user_id="u9")

    msgs = db.get_messages("thr_rt")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello there"
    assert msgs[1]["citations"] == [{"id": "chk_001"}]
