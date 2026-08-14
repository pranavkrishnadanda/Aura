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
def reset_inmemory_stores():
    """Isolate the in-memory fallback stores so tests cannot leak state into each other.

    Threads, messages and the ingest job table are module-level dicts. Without this,
    a thread created by one test is visible to the next, which hides real isolation
    bugs and makes failures order-dependent.
    """
    from app import db, ingest

    db._threads.clear()
    db._messages.clear()
    ingest._jobs.clear()
    yield
