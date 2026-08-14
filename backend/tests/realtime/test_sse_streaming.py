"""SSE wire-format and streaming-semantics tests for /api/v1/chat/stream.

LLM_PROVIDER defaults to "mock" so tokens are deterministic and no network call
is made. One test overrides the provider with a mocked stream_groq to exercise
adversarial token content (newlines, unicode, quotes) round-tripping through JSON.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

import app.rag as rag
from app.config import settings
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
    """Parse an SSE body into a list of (event_name, raw_data_str) tuples,
    preserving wire order and NOT pre-parsing JSON, so tests can assert on the
    raw framing as well as the decoded payload."""
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
        events.append((event_name, "\n".join(data_lines)))
    return events


def stream_raw(message, thread_id):
    AppStatus.should_exit_event = None
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": message, "thread_id": thread_id}
    ) as r:
        assert r.status_code == 200
        raw = r.read().decode()
    return raw


# ---------------------------------------------------------------------------
# Frame validity
# ---------------------------------------------------------------------------

def test_every_frame_is_valid_sse_with_json_data():
    raw = stream_raw("First-line therapy for hypertension with CKD?", "sse_frame_thread")
    events = parse_sse(raw)

    assert len(events) >= 3  # at least meta, one token, done
    for name, data_raw in events:
        assert name  # non-empty event name
        parsed = json.loads(data_raw)  # must not raise
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Ordering / time-to-first-token
# ---------------------------------------------------------------------------

def test_meta_event_arrives_first_and_carries_required_fields():
    raw = stream_raw("First-line therapy for hypertension with CKD?", "sse_meta_first_thread")
    events = parse_sse(raw)

    assert events[0][0] == "meta"
    meta = json.loads(events[0][1])
    assert "citations" in meta
    assert "is_refusal" in meta
    assert meta["thread_id"] == "sse_meta_first_thread"


def test_done_full_text_equals_concatenation_of_token_events():
    raw = stream_raw("First-line therapy for hypertension with CKD?", "sse_concat_thread")
    events = parse_sse(raw)

    token_texts = [json.loads(d)["token"] for name, d in events if name == "token"]
    done = json.loads(next(d for name, d in events if name == "done"))

    assert done["full_text"] == "".join(token_texts).strip()


def test_heartbeat_frame_follows_done():
    raw = stream_raw("Hi", "sse_heartbeat_thread")
    events = parse_sse(raw)

    names = [n for n, _ in events]
    assert names[-1] == "heartbeat"
    assert names[-2] == "done"
    heartbeat = json.loads(events[-1][1])
    assert "ts" in heartbeat


# ---------------------------------------------------------------------------
# Adversarial token content survives the JSON round trip
# ---------------------------------------------------------------------------

def test_tokens_with_newlines_unicode_and_quotes_survive_json_round_trip(monkeypatch):
    tricky_token = 'Line1\nLine2 "quoted" café \U0001F600 [1]'

    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")

    async def fake_stream_groq(prompt, context):
        yield tricky_token

    monkeypatch.setattr(rag, "stream_groq", fake_stream_groq)

    raw = stream_raw("First-line therapy for hypertension with CKD?", "sse_unicode_thread")
    events = parse_sse(raw)

    token_events = [json.loads(d)["token"] for name, d in events if name == "token"]
    assert token_events == [tricky_token]

    done = json.loads(next(d for name, d in events if name == "done"))
    assert done["full_text"] == tricky_token.strip()
