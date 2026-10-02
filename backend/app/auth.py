"""Request identity.

Two modes, selected by settings.ENABLE_AUTH:

- ENABLE_AUTH=false (default, demo): callers are anonymous and rate limiting is
  the only protection. Each browser sends a random per-visitor id in X-Anon-Id;
  a well-formed id is hashed into a stable "anon_..." user_id so one visitor's
  threads are scoped away from another's. A missing or malformed id falls back
  to the shared ANONYMOUS identity. X-API-Key never upgrades a caller to
  "authenticated" in this mode.
- ENABLE_AUTH=true: a valid X-API-Key from settings.API_KEYS is required and
  anything else is rejected with 401. X-Anon-Id is ignored, and identity is
  derived only from the matched configured key.

The anonymous id is an unauthenticated, client-chosen value: it separates
visitors who do not share it, but it is not a secret and proves nothing.
"""
import hashlib
import hmac
import logging
import re
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger("aura.auth")

ANONYMOUS = {"user_id": "anonymous", "tier": "anonymous"}

# Shape of the browser-generated id (a UUID in practice). Anything else is
# treated as absent rather than trusted as an identifier.
_ANON_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _valid_keys() -> list[str]:
    return [k.strip() for k in settings.API_KEYS.split(",") if k.strip()]


def _match_key(presented: str) -> Optional[str]:
    """Return the matching configured key, comparing in constant time."""
    for key in _valid_keys():
        if hmac.compare_digest(presented, key):
            return key
    return None


def _user_id_for(key: str) -> str:
    # Derive a stable, non-reversible id so the raw key never becomes the
    # identifier that gets stored on threads and written to logs.
    return "usr_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def _anon_identity(x_anon_id: object) -> dict:
    # isinstance guards direct calls that leave the FastAPI Header default in place.
    if isinstance(x_anon_id, str) and _ANON_ID_RE.fullmatch(x_anon_id):
        digest = hashlib.sha256(x_anon_id.encode()).hexdigest()[:16]
        return {"user_id": "anon_" + digest, "tier": "anonymous"}
    return ANONYMOUS


async def get_current_user(
    x_api_key: Optional[str] = Header(None),
    x_anon_id: Optional[str] = Header(None),
) -> dict:
    if not settings.ENABLE_AUTH:
        return _anon_identity(x_anon_id)

    configured = _valid_keys()
    if not configured:
        # Fail closed. Enabling auth with no keys configured must not degrade to
        # letting everyone through.
        logger.error("ENABLE_AUTH is true but API_KEYS is empty; rejecting all requests")
        raise HTTPException(status_code=503, detail="Authentication is misconfigured")

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")

    matched = _match_key(x_api_key)
    if not matched:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {"user_id": _user_id_for(matched), "tier": "authenticated"}
