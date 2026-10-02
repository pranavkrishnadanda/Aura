"""Unit tests for app.auth.

Two modes selected by settings.ENABLE_AUTH: anonymous demo mode where a header
must never grant "authenticated" status, and enforced mode where a missing
API_KEYS config must fail closed (503) rather than silently admit everyone.
"""
import pytest
from fastapi import HTTPException

from app.auth import (
    ANONYMOUS,
    _match_key,
    _user_id_for,
    _valid_keys,
    get_current_user,
)
from app.config import settings


# ---- _valid_keys ----

@pytest.mark.unit
def test_valid_keys_parses_comma_separated(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "key1,key2,key3")
    assert _valid_keys() == ["key1", "key2", "key3"]


@pytest.mark.unit
def test_valid_keys_strips_whitespace(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", " key1 , key2 ,key3 ")
    assert _valid_keys() == ["key1", "key2", "key3"]


@pytest.mark.unit
def test_valid_keys_ignores_blank_entries(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "key1,,  ,key2,")
    assert _valid_keys() == ["key1", "key2"]


@pytest.mark.unit
def test_valid_keys_empty_string_returns_empty_list(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "")
    assert _valid_keys() == []


# ---- _match_key ----

@pytest.mark.unit
def test_match_key_returns_matching_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "secretkey1,secretkey2")
    assert _match_key("secretkey2") == "secretkey2"


@pytest.mark.unit
def test_match_key_unknown_key_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "secretkey1,secretkey2")
    assert _match_key("not-a-configured-key") is None


@pytest.mark.unit
def test_match_key_empty_presented_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "secretkey1")
    assert _match_key("") is None


@pytest.mark.unit
def test_match_key_uses_constant_time_comparison(monkeypatch):
    """Regression guard: matching must go through hmac.compare_digest rather
    than == , so timing does not leak how many leading chars of a guess are
    correct. We assert the implementation calls compare_digest for every
    candidate rather than short-circuiting with a cheaper comparison."""
    import hmac as hmac_module

    monkeypatch.setattr(settings, "API_KEYS", "abc,defg")
    calls = []
    real_compare = hmac_module.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr("app.auth.hmac.compare_digest", spy)
    _match_key("defg")
    assert calls == [("defg", "abc"), ("defg", "defg")]


# ---- _user_id_for ----

@pytest.mark.unit
def test_user_id_for_is_stable_for_same_key():
    assert _user_id_for("mykey") == _user_id_for("mykey")


@pytest.mark.unit
def test_user_id_for_differs_per_key():
    assert _user_id_for("keyA") != _user_id_for("keyB")


@pytest.mark.unit
def test_user_id_for_is_non_reversible():
    """The raw key must never appear verbatim in the derived id."""
    key = "super-secret-api-key-value"
    result = _user_id_for(key)
    assert key not in result


@pytest.mark.unit
def test_user_id_for_has_expected_prefix_and_length():
    result = _user_id_for("anykey")
    assert result.startswith("usr_")
    assert len(result) == len("usr_") + 16


# ---- get_current_user ----

@pytest.mark.unit
async def test_get_current_user_auth_disabled_returns_anonymous_even_with_header(monkeypatch):
    """Regression: a header must not silently grant authenticated status when
    ENABLE_AUTH is false."""
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    result = await get_current_user(x_api_key="some-key-someone-sent")
    assert result == ANONYMOUS


@pytest.mark.unit
async def test_get_current_user_auth_disabled_no_header_returns_anonymous(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    result = await get_current_user(x_api_key=None)
    assert result == ANONYMOUS


@pytest.mark.unit
async def test_get_current_user_auth_enabled_empty_keys_fails_closed(monkeypatch):
    """Regression: enabling auth without configuring any keys must reject
    everyone with 503, not silently let every request through."""
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(x_api_key="anything")
    assert exc_info.value.status_code == 503


@pytest.mark.unit
async def test_get_current_user_missing_key_returns_401(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "goodkey")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(x_api_key=None)
    assert exc_info.value.status_code == 401


@pytest.mark.unit
async def test_get_current_user_bad_key_returns_401(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "goodkey")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(x_api_key="wrongkey")
    assert exc_info.value.status_code == 401


@pytest.mark.unit
async def test_get_current_user_good_key_returns_authenticated(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "goodkey")
    result = await get_current_user(x_api_key="goodkey")
    assert result["tier"] == "authenticated"
    assert result["user_id"] == _user_id_for("goodkey")


# ---- per-visitor anonymous identity (X-Anon-Id) ----

_VALID_ANON_ID = "3f2b8c1e-9a4d-4e6f-b1c2-7d8e9f0a1b2c"


@pytest.mark.unit
async def test_anon_id_yields_hashed_anonymous_identity(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    result = await get_current_user(x_api_key=None, x_anon_id=_VALID_ANON_ID)
    assert result["tier"] == "anonymous"
    user_id = result["user_id"]
    assert user_id.startswith("anon_")
    suffix = user_id[len("anon_"):]
    assert len(suffix) == 16
    assert all(c in "0123456789abcdef" for c in suffix)
    # The raw client id is hashed, never stored verbatim.
    assert _VALID_ANON_ID not in user_id


@pytest.mark.unit
async def test_anon_id_is_stable_for_same_id(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    first = await get_current_user(x_api_key=None, x_anon_id=_VALID_ANON_ID)
    second = await get_current_user(x_api_key=None, x_anon_id=_VALID_ANON_ID)
    assert first["user_id"] == second["user_id"]


@pytest.mark.unit
async def test_anon_id_differs_per_visitor(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    a = await get_current_user(x_api_key=None, x_anon_id="visitor-aaaa-1111")
    b = await get_current_user(x_api_key=None, x_anon_id="visitor-bbbb-2222")
    assert a["user_id"] != b["user_id"]
    assert a["user_id"] != ANONYMOUS["user_id"]
    assert b["user_id"] != ANONYMOUS["user_id"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_id",
    [
        "short",  # under 8 chars
        "has spaces in it",
        "path/traversal-id",
        "a" * 65,  # over 64 chars
        "",
    ],
)
async def test_malformed_anon_id_falls_back_to_anonymous(monkeypatch, bad_id):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    result = await get_current_user(x_api_key=None, x_anon_id=bad_id)
    assert result == ANONYMOUS


@pytest.mark.unit
async def test_anon_id_length_bounds_are_inclusive(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    for ok_id in ("a" * 8, "a" * 64):
        result = await get_current_user(x_api_key=None, x_anon_id=ok_id)
        assert result["user_id"].startswith("anon_")


@pytest.mark.unit
async def test_anon_id_ignored_when_auth_enabled_without_key(monkeypatch):
    """With auth enforced, X-Anon-Id must not stand in for an API key."""
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "goodkey")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(x_api_key=None, x_anon_id=_VALID_ANON_ID)
    assert exc_info.value.status_code == 401


@pytest.mark.unit
async def test_anon_id_ignored_when_auth_enabled_with_key(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEYS", "goodkey")
    result = await get_current_user(x_api_key="goodkey", x_anon_id=_VALID_ANON_ID)
    assert result == {"user_id": _user_id_for("goodkey"), "tier": "authenticated"}
