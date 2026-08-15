"""Provider-level tests for app.rag.generate_answer / stream_groq / stream_gemini.

Every provider SDK call is mocked -- these tests must never touch the network.
The configured GEMINI_API_KEY in this environment is real but out of credits, so
a live call would 429 and any test relying on network behavior would be flaky
by construction.
"""
import re
import pytest

import app.rag as rag
from app.config import settings
from app.rag import generate_answer, stream_gemini

pytestmark = pytest.mark.e2e


def make_chunk(text, doc_title="Doc", page=1, doc_id=None, cid=None):
    return {
        "id": cid or f"chk_{abs(hash((text, doc_title, page))) % 10**8}",
        "doc_id": doc_id or "doc_test",
        "doc_title": doc_title,
        "page": page,
        "text": text,
    }


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Belt-and-braces: never let a test in this module reach a real provider."""
    import app.ingest as ingest

    monkeypatch.setattr(rag, "_embed_query_gemini", lambda q: None)
    monkeypatch.setattr(ingest, "_embed_texts", lambda texts: [None] * len(texts))
    yield


async def _collect(agen):
    return [tok async for tok in agen]


# ---------------------------------------------------------------------------
# Providers replay a mocked stream, in order
# ---------------------------------------------------------------------------

async def test_groq_provider_yields_mocked_tokens_in_order(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")

    async def fake_stream_groq(prompt, context):
        for tok in ["Al", "pha ", "Be", "ta "]:
            yield tok

    monkeypatch.setattr(rag, "stream_groq", fake_stream_groq)

    retrieved = [(make_chunk("Alpha content is grounded.", "Doc A", 1), 0.9)]
    tokens = await _collect(generate_answer("what is alpha", retrieved))

    assert tokens == ["Al", "pha ", "Be", "ta "]


async def test_gemini_provider_yields_mocked_tokens_in_order(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")

    async def fake_stream_gemini(prompt, context):
        for tok in ["Gam", "ma ", "Del", "ta "]:
            yield tok

    monkeypatch.setattr(rag, "stream_gemini", fake_stream_gemini)

    retrieved = [(make_chunk("Gamma content is grounded.", "Doc B", 2), 0.9)]
    tokens = await _collect(generate_answer("what is gamma", retrieved))

    assert tokens == ["Gam", "ma ", "Del", "ta "]


# ---------------------------------------------------------------------------
# Provider failure mid-stream
# ---------------------------------------------------------------------------

async def test_provider_failure_mid_stream_reports_failure_not_a_fabricated_answer(monkeypatch):
    """Regression guard.

    generate_answer's exception handler used to fall back to echoing raw
    retrieved chunk text with a fabricated "[1]" appended, so a provider outage
    looked to the caller exactly like a normal grounded answer. It must instead
    say generation failed and name the sources it had retrieved, with no
    citation brackets attached to anything.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")

    async def fake_stream_groq_raises(prompt, context):
        yield "partial "
        raise RuntimeError("groq unavailable")

    monkeypatch.setattr(rag, "stream_groq", fake_stream_groq_raises)

    chunk_text = "Zebra finches exhibit vocal learning behavior distinct from other passerines."
    retrieved = [(make_chunk(chunk_text, "Doc A", 1), 0.9)]
    full = "".join(await _collect(generate_answer("tell me about zebra finches", retrieved)))

    assert "couldn't complete" in full
    assert "unavailable" in full
    assert "Doc A p.1" in full
    # The old bug: raw chunk text dressed up as an answer with a fake citation.
    assert chunk_text not in full
    assert "[1]" not in full


# ---------------------------------------------------------------------------
# Mock provider only replays retrieved text, never invented clinical claims
# ---------------------------------------------------------------------------

async def test_mock_provider_replays_only_retrieved_text_with_correct_markers():
    """Regression guard.

    The mock provider previously appended fixed clinical claims -- "contra-
    indications including angioedema and renal artery stenosis [2]" and
    lisinopril dosing advice -- to any answer with more than one citation,
    regardless of the question or what the sources actually said. It must only
    ever replay the retrieved chunk text, tagged with the citation it came from.
    """
    chunk1 = make_chunk("Enoxaparin dosing for VTE prophylaxis.", "Trial Doc", 5)
    chunk2 = make_chunk("Zolbetuximab is administered every 3 weeks.", "Oncology Doc", 9)
    retrieved = [(chunk1, 0.9), (chunk2, 0.8)]

    full = "".join(await _collect(generate_answer("unrelated clinical question", retrieved)))

    assert f"{chunk1['text']} [1]" in full
    assert f"{chunk2['text']} [2]" in full
    assert "angioedema and renal artery stenosis" not in full
    assert "Initiate at low dose and titrate" not in full


# ---------------------------------------------------------------------------
# Citation markers correspond to the chunk they came from
# ---------------------------------------------------------------------------

async def test_citation_markers_correspond_to_the_originating_chunk(monkeypatch):
    """Markers given to the model must be exactly the markers the UI can resolve.

    Regression: the context was numbered over the UNFILTERED retrieved list while
    skipping below-threshold chunks, so a gap appeared ([1], [3]) while the API
    numbered its citation payload contiguously ([1], [2]). The model then cited
    [3], which the UI had no entry for, and it rendered as dead un-clickable text
    in a product whose promise is that every citation opens its source.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    captured = {}

    async def fake_stream_groq(prompt, context):
        captured["context"] = context
        yield "ok"

    monkeypatch.setattr(rag, "stream_groq", fake_stream_groq)

    chunk_a = make_chunk("Chunk A content.", "Doc A", 1)
    chunk_b = make_chunk("Chunk B content.", "Doc B", 2)  # below threshold
    chunk_c = make_chunk("Chunk C content.", "Doc C", 3)
    retrieved = [(chunk_a, 0.9), (chunk_b, 0.01), (chunk_c, 0.95)]

    await _collect(generate_answer("query", retrieved))

    ctx = captured["context"]
    assert "[1] Chunk A content." in ctx
    assert "[2] Chunk C content." in ctx      # contiguous, not [3]
    assert "Chunk B content." not in ctx      # below threshold, excluded entirely
    assert "[3]" not in ctx

    # And the numbering must match what the API hands the UI.
    from app.rag import build_context, effective_threshold
    _, kept = build_context(retrieved, effective_threshold())
    ui_idx = [i for i, _ in enumerate(kept, 1)]
    model_markers = sorted({int(m) for m in re.findall(r"\[(\d+)\]", ctx)})
    assert model_markers == ui_idx, "model markers and UI citation indices diverged"

    # Retrieval scores must not leak into the prompt.
    assert "score=" not in ctx


# ---------------------------------------------------------------------------
# Below-threshold retrieval -> boundary text, not an answer
# ---------------------------------------------------------------------------

async def test_below_threshold_retrieval_yields_boundary_not_an_answer():
    chunk = make_chunk("Some unrelated clinical fact.", "Doc Z", 1)
    retrieved = [(chunk, 0.01)]  # well below both the tfidf (0.10) and pgvector (0.85) floors

    full = "".join(await _collect(generate_answer("obscure question", retrieved)))

    assert "outside what I can source" in full  # from rag.out_of_scope_message
    assert chunk["text"] not in full
    assert "[1]" not in full


# ---------------------------------------------------------------------------
# Greeting bypass
# ---------------------------------------------------------------------------

async def test_greeting_bypass_yields_greeting_with_no_citations():
    full = "".join(await _collect(generate_answer("hi", [])))

    assert "clinical intelligence assistant" in full
    assert "[1]" not in full
    assert "[" not in full


# ---------------------------------------------------------------------------
# stream_gemini tolerates chunks whose .text raises (safety-blocked chunks)
# ---------------------------------------------------------------------------

async def test_stream_gemini_skips_chunks_whose_text_property_raises(monkeypatch):
    """.text is a property on the SDK's response chunk that raises when a chunk
    carries no usable part (safety block, non-STOP finish reason). stream_gemini
    must skip those chunks rather than letting the exception kill the stream.
    """
    import google.generativeai as genai_mod

    class GoodChunk:
        def __init__(self, text):
            self._text = text

        @property
        def text(self):
            return self._text

    class RaisingChunk:
        @property
        def text(self):
            raise ValueError("blocked by safety filters")

    fake_response = [GoodChunk("Hello "), RaisingChunk(), GoodChunk("world")]

    class FakeModel:
        def __init__(self, *a, **kw):
            pass

        def generate_content(self, prompt, stream=True):
            return fake_response

    monkeypatch.setattr(genai_mod, "configure", lambda **kw: None)
    monkeypatch.setattr(genai_mod, "GenerativeModel", FakeModel)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key-for-test")

    tokens = await _collect(stream_gemini("prompt", "context"))

    assert tokens == ["Hello ", "world"]
