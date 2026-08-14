"""
Production auth: API key or JWT placeholder
For portfolio $0, use X-API-Key header or allow anonymous with rate limiting
"""
from fastapi import Header, HTTPException
from typing import Optional

async def get_current_user(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    # In prod, verify JWT or API key against DB
    # For now, allow anonymous but tag user_id for rate limiting
    if x_api_key:
        # TODO: validate against DB / Supabase auth
        return {"user_id": f"api_{x_api_key[:8]}", "tier": "authenticated"}
    if authorization and authorization.startswith("Bearer "):
        # TODO: verify JWT
        return {"user_id": "jwt_user", "tier": "authenticated"}
    return {"user_id": "anonymous", "tier": "anonymous"}
