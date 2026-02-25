"""
Tests for API rate limiting middleware behavior.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import config
from app.api.middleware import register_middleware
from app.api.rate_limit import rate_limiter


def _build_app(monkeypatch) -> FastAPI:
    rate_limiter.reset()
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 60)
    monkeypatch.setattr(config, "RATE_LIMIT_DEFAULT_PER_WINDOW", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_ASK_PER_WINDOW", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_EMBEDDINGS_PER_WINDOW", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_SYNC_PER_WINDOW", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_INDEXING_PER_WINDOW", 2)
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 1024 * 1024)

    app = FastAPI()
    register_middleware(app)

    @app.get("/private")
    async def private():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def test_rate_limit_returns_429_after_limit(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    headers = {"X-API-Key": "x" * 32}

    assert client.get("/private", headers=headers).status_code == 200
    assert client.get("/private", headers=headers).status_code == 200

    third = client.get("/private", headers=headers)
    assert third.status_code == 429
    assert "Retry-After" in third.headers
    assert third.json()["detail"] == "Rate limit exceeded"


def test_rate_limit_scopes_by_key(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)

    first = client.get("/health")
    assert first.status_code == 200
    assert first.headers.get("X-RateLimit-Limit") == "2"
    assert first.headers.get("X-RateLimit-Window") == "60"

    assert client.get("/health").status_code == 200
    third = client.get("/health")
    assert third.status_code == 429
