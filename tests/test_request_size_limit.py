"""
Tests for request payload size guardrails in API middleware.
"""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.config import config
from app.api.middleware import register_middleware


def _build_app(monkeypatch, *, max_request_bytes: int) -> FastAPI:
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", max_request_bytes)

    app = FastAPI()
    register_middleware(app)

    @app.post("/echo")
    async def echo(request: Request):
        payload = await request.json()
        return payload

    return app


def test_request_payload_too_large_returns_413(monkeypatch):
    app = _build_app(monkeypatch, max_request_bytes=80)
    client = TestClient(app)
    headers = {"X-API-Key": "x" * 32}

    response = client.post("/echo", json={"text": "a" * 200}, headers=headers)
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large"
    assert response.headers.get("X-Max-Request-Bytes") == "80"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_request_payload_within_limit_is_accepted(monkeypatch):
    app = _build_app(monkeypatch, max_request_bytes=1024)
    client = TestClient(app)
    headers = {"X-API-Key": "x" * 32}

    response = client.post("/echo", json={"text": "ok"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["text"] == "ok"
