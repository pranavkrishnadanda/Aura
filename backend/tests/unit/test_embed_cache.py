"""Unit tests for app.rag._prune_embed_cache.

The query-embedding cache used to retain expired entries (they were skipped on
read but never removed), so the dict grew without bound and kept raw clinical
queries resident for the process lifetime. _prune_embed_cache is the fix: drop
anything past EMBED_CACHE_TTL, then cap the survivors at _EMBED_CACHE_MAX,
evicting the oldest first.
"""
import time

import pytest

from app import rag
from app.rag import _prune_embed_cache


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    """Every test gets an isolated cache dict instead of the shared module one."""
    monkeypatch.setattr(rag, "_embed_cache", {})
    yield


@pytest.mark.unit
def test_expired_entries_are_removed(monkeypatch):
    monkeypatch.setattr(rag.settings, "EMBED_CACHE_TTL", 100)
    now = time.time()
    rag._embed_cache["expired query"] = ([0.1, 0.2], now - 200)
    rag._embed_cache["fresh query"] = ([0.3, 0.4], now - 10)
    _prune_embed_cache()
    assert "expired query" not in rag._embed_cache
    assert "fresh query" in rag._embed_cache


@pytest.mark.unit
def test_entry_exactly_at_ttl_boundary_is_removed(monkeypatch):
    """now - ts >= TTL is the removal condition (strictly-greater-or-equal)."""
    monkeypatch.setattr(rag.settings, "EMBED_CACHE_TTL", 100)
    now = time.time()
    rag._embed_cache["boundary"] = ([0.1], now - 100)
    _prune_embed_cache()
    assert "boundary" not in rag._embed_cache


@pytest.mark.unit
def test_no_entries_removed_when_all_fresh(monkeypatch):
    monkeypatch.setattr(rag.settings, "EMBED_CACHE_TTL", 3600)
    now = time.time()
    for i in range(5):
        rag._embed_cache[f"q{i}"] = ([0.0], now)
    _prune_embed_cache()
    assert len(rag._embed_cache) == 5


@pytest.mark.unit
def test_cache_capped_at_max_size(monkeypatch):
    monkeypatch.setattr(rag.settings, "EMBED_CACHE_TTL", 3600)
    monkeypatch.setattr(rag, "_EMBED_CACHE_MAX", 5)
    now = time.time()
    for i in range(10):
        rag._embed_cache[f"q{i}"] = ([0.0], now + i)
    _prune_embed_cache()
    assert len(rag._embed_cache) == 5


@pytest.mark.unit
def test_cache_eviction_removes_oldest_entries_first(monkeypatch):
    monkeypatch.setattr(rag.settings, "EMBED_CACHE_TTL", 3600)
    monkeypatch.setattr(rag, "_EMBED_CACHE_MAX", 3)
    now = time.time()
    # q0 is oldest, q4 is newest.
    for i in range(5):
        rag._embed_cache[f"q{i}"] = ([0.0], now + i)
    _prune_embed_cache()
    assert set(rag._embed_cache.keys()) == {"q2", "q3", "q4"}


@pytest.mark.unit
def test_prune_on_empty_cache_is_a_noop(monkeypatch):
    monkeypatch.setattr(rag.settings, "EMBED_CACHE_TTL", 3600)
    _prune_embed_cache()
    assert rag._embed_cache == {}
