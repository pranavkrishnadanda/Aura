"""Unit tests for app.rag: effective_threshold, retrieval_mode, is_greeting.

retrieval_mode decides whether a query actually gets dense pgvector search or
falls back to sparse TF-IDF/token-overlap, and effective_threshold picks the
score floor to match. Previously effective_threshold always returned the
TF-IDF number even in pgvector mode, so the documented 0.85 grounding gate was
never enforced -- these tests pin every branch down individually.
"""
import pytest

from app import rag
from app.rag import effective_threshold, retrieval_mode, is_greeting


def _drive(monkeypatch, *, db_available, gemini_key, has_embedded, has_sklearn=True,
           retrieval_threshold=0.85, tfidf_threshold=0.10):
    monkeypatch.setattr(rag, "is_db_available", lambda: db_available)
    monkeypatch.setattr(rag.settings, "GEMINI_API_KEY", gemini_key)
    monkeypatch.setattr(rag, "_has_embedded_chunks", lambda: has_embedded)
    monkeypatch.setattr(rag, "HAS_SKLEARN", has_sklearn)
    monkeypatch.setattr(rag.settings, "RETRIEVAL_THRESHOLD", retrieval_threshold)
    monkeypatch.setattr(rag.settings, "TFIDF_THRESHOLD", tfidf_threshold)


# ---- retrieval_mode ----

@pytest.mark.unit
def test_retrieval_mode_pgvector_when_db_key_and_embedded_chunks_present(monkeypatch):
    _drive(monkeypatch, db_available=True, gemini_key="k", has_embedded=True)
    assert retrieval_mode() == "pgvector"


@pytest.mark.unit
def test_retrieval_mode_tfidf_when_db_unavailable(monkeypatch):
    _drive(monkeypatch, db_available=False, gemini_key="k", has_embedded=True, has_sklearn=True)
    assert retrieval_mode() == "tfidf"


@pytest.mark.unit
def test_retrieval_mode_tfidf_when_no_gemini_key(monkeypatch):
    _drive(monkeypatch, db_available=True, gemini_key="", has_embedded=True, has_sklearn=True)
    assert retrieval_mode() == "tfidf"


@pytest.mark.unit
def test_retrieval_mode_tfidf_when_no_embedded_chunks(monkeypatch):
    _drive(monkeypatch, db_available=True, gemini_key="k", has_embedded=False, has_sklearn=True)
    assert retrieval_mode() == "tfidf"


@pytest.mark.unit
def test_retrieval_mode_token_overlap_when_sklearn_missing_and_not_pgvector(monkeypatch):
    _drive(monkeypatch, db_available=False, gemini_key="", has_embedded=False, has_sklearn=False)
    assert retrieval_mode() == "token-overlap"


# ---- effective_threshold ----

@pytest.mark.unit
def test_effective_threshold_returns_retrieval_threshold_in_pgvector_mode(monkeypatch):
    """Regression: this used to unconditionally return TFIDF_THRESHOLD."""
    _drive(monkeypatch, db_available=True, gemini_key="k", has_embedded=True,
           retrieval_threshold=0.85, tfidf_threshold=0.10)
    assert effective_threshold() == 0.85


@pytest.mark.unit
def test_effective_threshold_returns_tfidf_threshold_when_db_unavailable(monkeypatch):
    _drive(monkeypatch, db_available=False, gemini_key="k", has_embedded=True,
           retrieval_threshold=0.85, tfidf_threshold=0.10)
    assert effective_threshold() == 0.10


@pytest.mark.unit
def test_effective_threshold_returns_tfidf_threshold_when_no_gemini_key(monkeypatch):
    _drive(monkeypatch, db_available=True, gemini_key="", has_embedded=True,
           retrieval_threshold=0.85, tfidf_threshold=0.10)
    assert effective_threshold() == 0.10


@pytest.mark.unit
def test_effective_threshold_returns_tfidf_threshold_when_no_embedded_chunks(monkeypatch):
    _drive(monkeypatch, db_available=True, gemini_key="k", has_embedded=False,
           retrieval_threshold=0.85, tfidf_threshold=0.10)
    assert effective_threshold() == 0.10


@pytest.mark.unit
def test_effective_threshold_uses_current_configured_values(monkeypatch):
    """A different configured pair is reflected exactly, not hardcoded defaults."""
    _drive(monkeypatch, db_available=True, gemini_key="k", has_embedded=True,
           retrieval_threshold=0.42, tfidf_threshold=0.99)
    assert effective_threshold() == 0.42


# ---- is_greeting ----

@pytest.mark.unit
@pytest.mark.parametrize("greeting", ["hi", "hello", "hey", "hiya", "help", "hi there", "hello there"])
def test_is_greeting_recognizes_known_greetings(greeting):
    assert is_greeting(greeting) is True


@pytest.mark.unit
def test_is_greeting_case_insensitive_and_trims_whitespace():
    assert is_greeting("   HELLO   ") is True


@pytest.mark.unit
def test_is_greeting_mixed_case_with_internal_text():
    assert is_greeting("Hi There") is True


@pytest.mark.unit
def test_is_greeting_false_for_clinical_question():
    assert is_greeting("What is first-line therapy for hypertension with CKD?") is False


@pytest.mark.unit
def test_is_greeting_false_for_empty_string():
    assert is_greeting("") is False


@pytest.mark.unit
def test_is_greeting_false_for_whitespace_only():
    assert is_greeting("   ") is False


@pytest.mark.unit
def test_is_greeting_false_for_greeting_word_embedded_in_longer_text():
    assert is_greeting("hello, can you help me with dosing for lisinopril") is False
