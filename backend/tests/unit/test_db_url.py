"""Unit tests for app.db._normalize_db_url.

A bare postgresql:// URL resolves to the psycopg2 dialect, which is not shipped
(only psycopg3 is in requirements.txt). On a clean install that raises
ModuleNotFoundError inside _init_engine's broad except, silently dropping the
whole app to the in-memory fallback with no DB at all. _normalize_db_url pins
the driver to postgresql+psycopg so that never happens.
"""
import pytest

from app.db import _normalize_db_url


@pytest.mark.unit
def test_bare_postgresql_scheme_gets_psycopg_driver():
    url = "postgresql://user:pass@localhost:5432/aura"
    assert _normalize_db_url(url) == "postgresql+psycopg://user:pass@localhost:5432/aura"


@pytest.mark.unit
def test_legacy_postgres_scheme_gets_psycopg_driver():
    """Heroku/Supabase-style postgres:// must also be normalized."""
    url = "postgres://user:pass@db.supabase.co:5432/postgres"
    assert _normalize_db_url(url) == "postgresql+psycopg://user:pass@db.supabase.co:5432/postgres"


@pytest.mark.unit
def test_already_qualified_psycopg_url_unchanged():
    url = "postgresql+psycopg://user:pass@localhost:5432/aura"
    assert _normalize_db_url(url) == url


@pytest.mark.unit
def test_already_qualified_psycopg2_url_unchanged():
    """An explicit psycopg2 driver choice is left alone, not silently upgraded."""
    url = "postgresql+psycopg2://user:pass@localhost:5432/aura"
    assert _normalize_db_url(url) == url


@pytest.mark.unit
def test_non_postgres_url_unchanged():
    url = "sqlite:///./local.db"
    assert _normalize_db_url(url) == url


@pytest.mark.unit
def test_empty_string_unchanged():
    assert _normalize_db_url("") == ""


@pytest.mark.unit
def test_only_first_occurrence_of_scheme_is_replaced():
    """count=1 replace: a scheme string appearing again later (e.g. in a query
    param) must not also get rewritten."""
    url = "postgresql://user:pass@localhost/db?options=postgresql://nested"
    result = _normalize_db_url(url)
    assert result.startswith("postgresql+psycopg://user:pass@localhost/db?options=")
    # The nested occurrence inside the query string stays untouched.
    assert "options=postgresql://nested" in result


# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_env_file_paths_are_absolute_and_cover_both_locations():
    """Settings must find .env regardless of the working directory.

    env_file was the bare relative string ".env", which resolves against the
    process's working directory. The documented run command is
    `cd backend && uvicorn ...`, so the repo-root .env was never read: keys were
    plainly present in the file and arrived as empty strings.
    """
    import os
    from app.config import Settings, _BACKEND_DIR, _REPO_ROOT

    paths = Settings.Config.env_file
    assert isinstance(paths, tuple), "a single path cannot cover both layouts"
    for p in paths:
        assert os.path.isabs(p), f"{p} is relative and depends on the cwd"
    assert str(_BACKEND_DIR / ".env") in paths
    assert str(_REPO_ROOT / ".env") in paths
