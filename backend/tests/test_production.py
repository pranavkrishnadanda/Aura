import pytest
from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def stream_text(msg, tid="t"):
    with client.stream("POST","/api/v1/chat/stream", json={"message":msg,"thread_id":tid}) as r:
        toks=[]
        lines=[]
        for line in r.iter_lines():
            lines.append(line)
            if '"token"' in line:
                try:
                    toks.append(json.loads(line.split("data:",1)[1])["token"])
                except: pass
        return "".join(toks), "\n".join(lines)

def test_greeting_not_refusal():
    txt,_ = stream_text("Hi","greet_prod")
    assert "clinical intelligence assistant" in txt
    assert "I cannot find" not in txt

def test_boundary_hair():
    txt,_ = stream_text("What do you know about hair problems","hair_prod")
    assert "outside my current clinical intelligence scope" in txt
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
    assert "[1]" in txt or "outside my current" not in txt  # should cite, not boundary

def test_health():
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"
    assert "threshold" in r.json()

def test_ready():
    r=client.get("/ready")
    assert r.status_code==200
    assert r.json()["ready"]==True

def test_upload_validation():
    r=client.post("/api/v1/documents/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code==400
