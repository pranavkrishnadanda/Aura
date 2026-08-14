"""Integration tests for CORS configuration and per-route rate limiting.

Both are baked into app.main at import time (the CORSMiddleware origin list, and
the rate-limit strings passed to each @limiter.limit(...) decorator are evaluated
once when the module's top-level code runs). To exercise non-default values
without touching application code, these tests monkeypatch app.config.settings
and then importlib.reload(app.main) to re-run that top-level code, always
restoring the previous settings + reloading again before the test ends so later
test files see the real default app.
"""
import importlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings

pytestmark = pytest.mark.integration


def _reload_with(monkeypatch, **overrides):
    originals = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        monkeypatch.setattr(settings, k, v)
    import app.main as main_module

    importlib.reload(main_module)
    return main_module, originals


def _restore(monkeypatch, originals):
    for k, v in originals.items():
        monkeypatch.setattr(settings, k, v)
    import app.main as main_module

    importlib.reload(main_module)


# ---- CORS ----

def test_cors_echoes_exact_origin_and_allows_credentials_when_configured(monkeypatch):
    main_module, originals = _reload_with(monkeypatch, CORS_ORIGINS="https://app.example.com")
    try:
        client = TestClient(main_module.app)
        r = client.get("/health", headers={"Origin": "https://app.example.com"})
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"
        assert r.headers.get("access-control-allow-credentials") == "true"
    finally:
        _restore(monkeypatch, originals)


def test_cors_wildcard_without_credentials_when_origins_unset(monkeypatch):
    """Regression: an unset CORS_ORIGINS must fall back to "*" WITHOUT also
    sending Allow-Credentials: true -- every browser rejects that exact header
    pairing outright, which made the API unreachable from the frontend with only
    an opaque CORS error to debug.
    """
    main_module, originals = _reload_with(monkeypatch, CORS_ORIGINS="")
    try:
        client = TestClient(main_module.app)
        r = client.get("/health", headers={"Origin": "https://anything.example.com"})
        assert r.headers.get("access-control-allow-origin") == "*"
        assert r.headers.get("access-control-allow-credentials") != "true"
    finally:
        _restore(monkeypatch, originals)


# ---- Rate limiting ----

def test_exceeding_limit_returns_429_with_detail_body(monkeypatch):
    main_module, originals = _reload_with(monkeypatch, RATE_LIMIT_ANON="1/minute")
    try:
        client = TestClient(main_module.app)
        r1 = client.get("/api/v1/threads")
        assert r1.status_code == 200

        r2 = client.get("/api/v1/threads")
        assert r2.status_code == 429
        assert r2.json().get("detail")
    finally:
        _restore(monkeypatch, originals)


def test_legacy_chat_stream_alias_is_rate_limited(monkeypatch):
    """Regression: /api/chat/stream (kept for older clients) previously carried no
    rate limit of its own while /api/v1/chat/stream did, so the limit on the most
    expensive endpoint in the app was bypassable simply by dropping /v1 from the
    URL.
    """
    main_module, originals = _reload_with(monkeypatch, RATE_LIMIT_ANON="1/minute")
    try:
        client = TestClient(main_module.app)
        body = {"message": "hi", "thread_id": "rl_legacy"}  # "hi" -> fast greeting path

        # First call opens an SSE stream; only need the status, so don't drain it.
        with client.stream("POST", "/api/chat/stream", json=body) as r1:
            assert r1.status_code == 200

        r2 = client.post("/api/chat/stream", json=body)
        assert r2.status_code == 429
        assert r2.json().get("detail")
    finally:
        _restore(monkeypatch, originals)
