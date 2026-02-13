"""
Common API dependencies: authentication, postgres guards, etc.
"""

from fastapi import HTTPException
from app.config import config


def require_postgres() -> None:
    """Raise 503 if PostgreSQL is not configured."""
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured. Set DATABASE_URL or POSTGRES_URL environment variable.",
        )
