"""Failure-path tests for /api/v1/chat/stream: mid-stream provider errors,
client disconnects, and retrieval outages.

app.main imports `generate_answer` and `retrieve_async` by name
(`from app.rag import retrieve_async, generate_answer, ...`), so overriding
their behavior for the endpoint requires patching the bound names on
app.main, not on app.rag -- patching app.rag.generate_answer would leave
app.main's already-imported reference untouched.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

import app.main as main
import app.rag as rag
from app.main import app

pytestmark = pytest.mark.realtime

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import app.ingest as ingest

    monkeypatch.setattr(rag, "_embed_query_gemini", lambda q: None)
    monkeypatch.setattr(ingest, "_embed_texts", lambda texts: [None] * len(texts))
    yield


def parse_sse(raw_text: str):
    events = []
    for block in raw_text.replace("\r\n", "\n").split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if event_name is None:
            continue
        data_raw = "\n".join(data_lines)
        events.append((event_name, json.loads(data_raw) if data_raw else None))
    return events


def stream_events(message, thread_id):
    AppStatus.should_exit_event = None
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": message, "thread_id": thread_id}
    ) as r:
        status = r.status_code
        content_type = r.headers.get("content-type", "")
        raw = r.read().decode()
    return status, content_type, parse_sse(raw)


# ---------------------------------------------------------------------------
# Provider failure mid-stream
# ---------------------------------------------------------------------------

def test_generation_failure_mid_stream_emits_error_event_and_terminates(monkeypatch):
    async def failing_generate_answer(query, retrieved):
        yield "partial answer text "
        raise RuntimeError("boom: provider exploded")

    monkeypatch.setattr(main, "generate_answer", failing_generate_answer)

    status, content_type, events = stream_events(
        "First-line therapy for hypertension with CKD?", "fail_midstream_thread"
    )

    assert status == 200
    names = [n for n, _ in events]
    assert "error" in names
    error_data = dict(events)["error"]
    assert "boom: provider exploded" in error_data["detail"]

    # The stream must still terminate cleanly -- done and heartbeat still fire,
    # proving the generator did not just die without closing out the response.
    assert names[-2:] == ["done", "heartbeat"]


def test_partial_stream_before_failure_is_still_persisted(monkeypatch):
    async def failing_generate_answer(query, retrieved):
        yield "first "
        yield "second "
        raise RuntimeError("provider outage")

    monkeypatch.setattr(main, "generate_answer", failing_generate_answer)

    tid = "fail_partial_persist_thread"
    status, _, events = stream_events("First-line therapy for hypertension with CKD?", tid)
    assert status == 200
    done = dict(events)["done"]
    assert done["full_text"] == "first second"

    msgs = client.get(f"/api/v1/threads/{tid}/messages").json()
    assistant_msg = next(m for m in msgs if m["role"] == "assistant")
    assert assistant_msg["content"] == "first second"


# ---------------------------------------------------------------------------
# Client disconnect
# ---------------------------------------------------------------------------

def test_client_disconnect_stops_generator_early(monkeypatch):
    """The endpoint checks request.is_disconnected() after every yielded
    token. If that check reports true immediately, the generator must stop
    after (at most) the first token instead of running the mock provider's
    full multi-token greeting to completion -- and must not hang.
    """
    async def always_disconnected(self):
        return True

    monkeypatch.setattr("starlette.requests.Request.is_disconnected", always_disconnected)

    status, _, events = stream_events("Hi", "disconnect_thread")
    assert status == 200

    names = [n for n, _ in events]
    token_count = names.count("token")
    # The greeting is dozens of tokens long when allowed to run to completion;
    # an immediate disconnect must cut that short.
    assert token_count <= 1
    # The response still closes out normally rather than hanging.
    assert names[-1] == "heartbeat"
    assert "done" in names


# ---------------------------------------------------------------------------
# Retrieval failure fails BEFORE the stream opens
# ---------------------------------------------------------------------------

def test_retrieval_failure_returns_503_before_stream_opens(monkeypatch):
    async def failing_retrieve_async(query, top_k=5):
        raise RuntimeError("knowledge base down")

    monkeypatch.setattr(main, "retrieve_async", failing_retrieve_async)

    # Not a greeting, so the endpoint must reach retrieval.
    r = client.post(
        "/api/v1/chat/stream",
        json={"message": "First-line therapy for hypertension with CKD?", "thread_id": "retrieval_fail_thread"},
    )

    assert r.status_code == 503
    content_type = r.headers.get("content-type", "")
    assert not content_type.startswith("text/event-stream")
    assert "temporarily unavailable" in r.json()["detail"]
