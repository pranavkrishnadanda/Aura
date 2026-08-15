import pytest
from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def stream_text(msg, tid="t"):
    # TestClient runs each request on a fresh event loop, while sse-starlette caches
    # its exit event process-globally. Reset it per request so a test may stream more
    # than once. Not needed in production, where uvicorn owns one loop for the
    # process lifetime -- see tests/conftest.py for the full explanation.
    from sse_starlette.sse import AppStatus
    AppStatus.should_exit_event = None

    with client.stream("POST","/api/v1/chat/stream", json={"message":msg,"thread_id":tid}) as r:
        toks=[]
        lines=[]
        for line in r.iter_lines():
            lines.append(line)
            if '"token"' in line:
                # Surface parse failures instead of swallowing them -- a bare
                # `except: pass` here would turn a broken stream into an empty
                # string and produce a confusing assertion error downstream.
                payload = line.split("data:",1)[1]
                toks.append(json.loads(payload)["token"])
        return "".join(toks), "\n".join(lines)

def test_greeting_not_refusal():
    txt,_ = stream_text("Hi","greet_prod")
    assert "clinical intelligence assistant" in txt
    assert "I cannot find" not in txt

def test_boundary_hair():
    txt,_ = stream_text("What do you know about hair problems","hair_prod")
    assert "outside what I can source" in txt  # from rag.out_of_scope_message
    assert "[1]" not in txt

def test_clinical_citation():
    txt,_ = stream_text("What is first-line therapy for hypertension with CKD?","clin_prod")
    assert "[1]" in txt
    assert "ACE inhibitors" in txt

def test_vague_hypertension_cites():
    txt,_ = stream_text("What do you know about hypertension","vague_prod")
    assert "[1]" in txt

def test_followup_anaphora():
    tid="follow_prod"
    stream_text("First-line therapy for hypertension with CKD?",tid)
    stream_text("What do you know about hair problems",tid)  # should not pollute
    txt,_ = stream_text("Are there contraindications for that dosage?",tid)
    assert "[1]" in txt or "outside what I can source" not in txt  # should cite, not boundary

def test_health_reports_storage_mode():
    r=client.get("/health")
    assert r.status_code==200
    body=r.json()
    # /health always answers, but must tell the truth about whether the DB is live.
    assert body["status"] in ("ok","degraded")
    assert "threshold" in body
    assert body["status"]=="ok" if body["pg_reachable"] else body["status"]=="degraded"
    # A degraded instance must name the cause instead of hiding it.
    if not body["pg_reachable"]:
        assert body["db_error"]
        assert body["storage_mode"]=="in-memory (ephemeral)"

def test_ready_fails_closed_when_db_down():
    r=client.get("/ready")
    body=r.json()
    # Readiness must be able to say "no" -- it previously returned a hardcoded True,
    # so the probe (and this test) could never fail.
    if body["pg"] and body["llm"]:
        assert r.status_code==200 and body["ready"] is True
    else:
        assert r.status_code==503 and body["ready"] is False

def test_upload_validation():
    r=client.post("/api/v1/documents/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code==400
