"""
API middleware: authentication, CORS, etc.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import config

# Endpoints that never require authentication
PUBLIC_ENDPOINTS = {"/health", "/config", "/docs", "/redoc", "/openapi.json"}


async def api_key_middleware(request, call_next):
    """
    Verify API key for all endpoints except public ones.
    If BRAIN_API_KEY is not configured, all requests are allowed (dev mode).
    """
    if request.url.path in PUBLIC_ENDPOINTS:
        return await call_next(request)

    if not config.BRAIN_API_KEY:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing API key. Include 'X-API-Key' header."},
        )

    if api_key != config.BRAIN_API_KEY:
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid API key"},
        )

    return await call_next(request)


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI app."""
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key authentication
    app.middleware("http")(api_key_middleware)
