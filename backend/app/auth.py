"""Request identity.

Two modes, selected by settings.ENABLE_AUTH:

- ENABLE_AUTH=false (default, demo): everyone is the same anonymous caller and
  rate limiting is the only protection. Callers are NOT tagged as authenticated
  just because they sent a header.
- ENABLE_AUTH=true: a valid X-API-Key from settings.API_KEYS is required and
  anything else is rejected with 401.

The previous implementation returned tier="authenticated" with a user_id derived
from whatever X-API-Key the caller supplied, and treated any "Bearer ..." value
as a logged-in user, without validating either. Since user_id is what scopes a
caller's threads, any client could adopt another user's identity by guessing the
first eight characters of their key, and ENABLE_AUTH was never consulted at all.
"""
import hashlib
import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger("aura.auth")

ANONYMOUS = {"user_id": "anonymous", "tier": "anonymous"}


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


async def get_current_user(x_api_key: Optional[str] = Header(None)):
    if not settings.ENABLE_AUTH:
        return ANONYMOUS

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
