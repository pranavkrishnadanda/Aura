import pytest


@pytest.fixture(autouse=True)
def reset_sse_app_status():
    """Reset sse-starlette's process-global exit event between tests.

    EventSourceResponse lazily caches ``AppStatus.should_exit_event`` as a class
    attribute (sse_starlette/sse.py:233). TestClient spins up a fresh event loop
    per request, so the event created during the first streaming test stays bound
    to that first loop and every later streaming test dies with
    "RuntimeError: <asyncio.locks.Event ...> is bound to a different event loop".

    That is why the streaming tests passed individually but failed as a suite.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None


@pytest.fixture(autouse=True)
def pin_provider_for_tests(monkeypatch):
    """Force the offline provider regardless of what .env or the shell says.

    Without this the suite's behaviour depends on ambient configuration: adding a
    real key to .env silently switched every generation to a live API call, so
    tests that assert on the mock provider's exact output started failing, and the
    suite began costing quota and network flakiness. A test that wants a real or
    fake provider sets it explicitly.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    yield


@pytest.fixture(autouse=True)
def reset_inmemory_stores():
    """Isolate the in-memory fallback stores so tests cannot leak state into each other.

    Threads, messages, chunks and the ingest job table are all module-level dicts.
    Without this, a thread created by one test is visible to the next, which hides
    real isolation bugs and makes failures order-dependent.

    The chunk store is restored to the seeded baseline rather than emptied: the
    retrieval tests rely on the SEED_CHUNKS clinical guidelines being present, while
    the ingest tests add to it. Leaving those additions behind made corpus-size
    assertions pass alone and fail in a full run.
    """
    from app import db, ingest

    db._threads.clear()
    db._messages.clear()
    ingest._jobs.clear()
    db._chunks.clear()
    db._chunks.update({c["id"]: dict(c) for c in db.SEED_CHUNKS})
    _reset_database(db)
    yield


def _reset_database(db) -> None:
    """Truncate and re-seed the real tables when a database is attached.

    Clearing only the in-memory dicts is enough locally, where there is no
    database, but under CI the suite runs against real Postgres and every row
    written by one test survives into the next -- so corpus-count assertions pass
    in isolation and fail in a full run. Storage backend must not change test
    outcomes.
    """
    if not db.is_db_available() or not db._SessionLocal:
        return
    from sqlalchemy import text as sql_text

    try:
        with db._SessionLocal() as s:
            s.execute(sql_text("TRUNCATE messages, threads, chunks RESTART IDENTITY CASCADE"))
            for c in db.SEED_CHUNKS:
                s.execute(
                    sql_text(
                        "INSERT INTO chunks (id, doc_id, doc_title, page, text) "
                        "VALUES (:id, :doc_id, :doc_title, :page, :text)"
                    ),
                    c,
                )
            s.commit()
    except Exception as e:  # pragma: no cover - diagnostic path
        import logging

        logging.getLogger("tests").warning("database reset failed: %s", e)
