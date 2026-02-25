"""
Tests for /security/self-check endpoint behavior.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import config
from app.api.middleware import register_middleware
from app.api.routes.security import router as security_router


def _make_app(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "PUBLIC_API_MODE", True)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "DEBUG", False)
    monkeypatch.setattr(config, "API_HOST", "127.0.0.1")

    app = FastAPI()
    register_middleware(app)
    app.include_router(security_router)
    return app


def test_security_endpoint_requires_api_key(monkeypatch):
    app = _make_app(monkeypatch)
    client = TestClient(app)
    response = client.get("/security/self-check")
    assert response.status_code == 401


def test_security_endpoint_returns_report(monkeypatch):
    app = _make_app(monkeypatch)
    client = TestClient(app)
    response = client.get("/security/self-check", headers={"X-API-Key": "x" * 32})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "public"
    assert data["fail_fast"] is True
    assert "checks" in data
