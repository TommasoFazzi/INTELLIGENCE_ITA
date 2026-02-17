"""Shared API authentication module."""
import os
import secrets
import logging

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

INTELLIGENCE_API_KEY = os.getenv("INTELLIGENCE_API_KEY")

if not INTELLIGENCE_API_KEY:
    logger.warning(
        "INTELLIGENCE_API_KEY not set — API authentication is DISABLED. "
        "Set the env var for production. Generate one with: "
        'python -c "import secrets; print(secrets.token_urlsafe(32))"'
    )


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key for protected endpoints."""
    if not INTELLIGENCE_API_KEY:
        # Development mode: allow access but warn
        return "dev_mode"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header.",
        )

    if not secrets.compare_digest(api_key, INTELLIGENCE_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key
