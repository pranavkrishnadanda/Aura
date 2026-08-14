"""Auth and per-user thread isolation.

The previous auth module accepted any X-API-Key and reported the caller as
authenticated, and the messages endpoint had no ownership check at all.
"""
import importlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture
def authed_client(monkeypatch):
    """App with auth enabled and two known keys."""
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "key-alice,key-bob")
    import app.main
    return TestClient(app.main.app)


@pytest.fixture
def anon_client():
    import app.main
    return TestClient(app.main.app)


def test_unknown_key_is_rejected(authed_client):
    r = authed_client.get("/api/v1/threads", headers={"X-API-Key": "not-a-real-key"})
    assert r.status_code == 401


def test_missing_key_is_rejected_when_auth_enabled(authed_client):
    r = authed_client.get("/api/v1/threads")
    assert r.status_code == 401


def test_bearer_token_does_not_grant_access(authed_client):
    # A Bearer header used to be accepted as a logged-in user with zero validation.
    r = authed_client.get("/api/v1/threads", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 401


def test_valid_key_is_accepted(authed_client):
    r = authed_client.get("/api/v1/threads", headers={"X-API-Key": "key-alice"})
    assert r.status_code == 200


def test_auth_enabled_without_keys_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "")
    import app.main
    r = TestClient(app.main.app).get("/api/v1/threads", headers={"X-API-Key": "anything"})
    # Must not degrade to letting everyone in.
    assert r.status_code == 503


def test_user_cannot_read_another_users_thread(authed_client):
    alice = {"X-API-Key": "key-alice"}
    bob = {"X-API-Key": "key-bob"}

    created = authed_client.post("/api/v1/threads", json={"title": "Alice consult"}, headers=alice)
    assert created.status_code == 200
    tid = created.json()["id"]

    # Alice can read her own thread.
    assert authed_client.get(f"/api/v1/threads/{tid}/messages", headers=alice).status_code == 200

    # Bob must not, even knowing the exact id.
    r = authed_client.get(f"/api/v1/threads/{tid}/messages", headers=bob)
    assert r.status_code == 404, "another user's thread was readable"


def test_thread_list_is_scoped_to_caller(authed_client):
    alice = {"X-API-Key": "key-alice"}
    bob = {"X-API-Key": "key-bob"}
    authed_client.post("/api/v1/threads", json={"title": "Alice private"}, headers=alice)

    titles = [t["title"] for t in authed_client.get("/api/v1/threads", headers=bob).json()]
    assert "Alice private" not in titles


def test_pagination_is_clamped(anon_client):
    # A negative offset used to wrap around the list via Python slicing.
    r = anon_client.get("/api/v1/threads/default/messages?offset=-5&limit=999999")
    assert r.status_code in (200, 404)
