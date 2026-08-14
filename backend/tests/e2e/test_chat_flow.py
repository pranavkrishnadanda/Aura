"""Full end-to-end chat flow through the real FastAPI app via TestClient.

LLM_PROVIDER defaults to "mock" so these tests exercise the real HTTP surface,
thread/message persistence and retrieval, without ever calling a real LLM or
embedding provider.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

import app.rag as rag
from app.main import app

pytestmark = pytest.mark.e2e

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import app.ingest as ingest

    monkeypatch.setattr(rag, "_embed_query_gemini", lambda q: None)
    monkeypatch.setattr(ingest, "_embed_texts", lambda texts: [None] * len(texts))
    yield


def parse_sse(raw_text: str):
    """Parse an SSE body ("event: X\\r\\ndata: {...}\\r\\n\\r\\n"...) into
    a list of (event_name, parsed_json) tuples, in wire order."""
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


def send_chat(message, thread_id):
    # TestClient runs each request on a fresh event loop while sse-starlette
    # caches its exit event process-globally; reset it so a test may stream
    # more than once. See tests/conftest.py for the full explanation.
    AppStatus.should_exit_event = None
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": message, "thread_id": thread_id}
    ) as r:
        assert r.status_code == 200
        content_type = r.headers.get("content-type", "")
        raw = r.read().decode()
    return content_type, parse_sse(raw)


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------

def test_stream_response_is_event_stream_with_meta_token_done_sequence():
    content_type, events = send_chat("Hi", "shape_thread")

    assert content_type.startswith("text/event-stream")
    names = [e[0] for e in events]
    assert names[0] == "meta"
    assert "token" in names
    assert "done" in names
    # every token event must come after meta and before done
    meta_idx = names.index("meta")
    done_idx = names.index("done")
    token_indices = [i for i, n in enumerate(names) if n == "token"]
    assert all(meta_idx < i < done_idx for i in token_indices)


# ---------------------------------------------------------------------------
# Thread creation regression
# ---------------------------------------------------------------------------

def test_referenced_thread_is_created_under_that_exact_id_and_only_once():
    """Regression guard.

    create_thread's positional signature is (title, ...). The chat endpoint used
    to call create_thread(body.thread_id) -- passing the client's thread id as
    the *title* -- so a fresh, unrelated random id was created instead, the
    referenced thread never existed, and every later add_message() to that
    referenced id silently created yet another, untitled, anonymous thread.
    """
    tid = "custom_thread_regression_1"
    send_chat("Hi", tid)

    threads = client.get("/api/v1/threads").json()
    matching = [t for t in threads if t["id"] == tid]
    assert len(matching) == 1, f"expected exactly one thread with id={tid!r}, found {len(matching)}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_user_and_assistant_messages_are_persisted_and_retrievable():
    tid = "persist_thread_1"
    query = "First-line therapy for hypertension with CKD?"
    _, events = send_chat(query, tid)
    done_data = dict(events)["done"]

    msgs = client.get(f"/api/v1/threads/{tid}/messages").json()
    roles = [m["role"] for m in msgs]
    assert "user" in roles and "assistant" in roles

    user_msg = next(m for m in msgs if m["role"] == "user")
    assistant_msg = next(m for m in msgs if m["role"] == "assistant")
    assert user_msg["content"] == query
    assert assistant_msg["content"] == done_data["full_text"]


def test_meta_citations_match_done_citations():
    _, events = send_chat("First-line therapy for hypertension with CKD?", "citations_match_thread")
    data = dict(events)
    assert data["meta"]["citations"] == data["done"]["citations"]


# ---------------------------------------------------------------------------
# Grounded vs out-of-scope
# ---------------------------------------------------------------------------

def test_clinical_seeded_query_returns_grounded_cited_answer():
    _, events = send_chat("First-line therapy for hypertension with CKD?", "clinical_thread")
    full_text = dict(events)["done"]["full_text"]
    assert "[1]" in full_text
    assert "ACE inhibitors" in full_text


def test_off_topic_query_returns_boundary_text_with_no_citation():
    _, events = send_chat("What do you know about hair problems", "offtopic_thread")
    full_text = dict(events)["done"]["full_text"]
    assert "outside my current clinical intelligence scope" in full_text
    assert "[1]" not in full_text


# ---------------------------------------------------------------------------
# Thread isolation
# ---------------------------------------------------------------------------

def test_two_threads_keep_separate_histories():
    tid_a = "iso_thread_a"
    tid_b = "iso_thread_b"
    send_chat("First-line therapy for hypertension with CKD?", tid_a)
    send_chat("What do you know about hair problems", tid_b)

    msgs_a = client.get(f"/api/v1/threads/{tid_a}/messages").json()
    msgs_b = client.get(f"/api/v1/threads/{tid_b}/messages").json()

    contents_a = [m["content"] for m in msgs_a]
    contents_b = [m["content"] for m in msgs_b]

    assert "What do you know about hair problems" not in contents_a
    assert "First-line therapy for hypertension with CKD?" not in contents_b
    assert len(msgs_a) == 2  # one user, one assistant
    assert len(msgs_b) == 2
