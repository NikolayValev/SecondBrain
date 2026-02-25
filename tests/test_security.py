"""
Security-focused tests for API middleware and path handling.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import config
from app.api.middleware import register_middleware
from app.services.file_service import _normalize_relative_path


@pytest.fixture
def middleware_app(monkeypatch):
    """Create a minimal app instance with production-like middleware."""
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "a" * 32)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)

    app = FastAPI()
    register_middleware(app)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/private")
    async def private():
        return {"ok": True}

    @app.get("/config")
    async def config_endpoint():
        return {"ok": True}

    return app


def test_private_endpoint_requires_api_key(middleware_app):
    client = TestClient(middleware_app)
    response = client.get("/private")
    assert response.status_code == 401


def test_private_endpoint_rejects_invalid_key(middleware_app):
    client = TestClient(middleware_app)
    response = client.get("/private", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403


def test_public_health_stays_accessible(middleware_app):
    client = TestClient(middleware_app)
    response = client.get("/health")
    assert response.status_code == 200


def test_config_endpoint_is_protected_by_default(middleware_app):
    client = TestClient(middleware_app)
    response = client.get("/config")
    assert response.status_code == 401


def test_security_headers_are_added(middleware_app):
    client = TestClient(middleware_app)
    response = client.get("/health")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "no-referrer"


def test_normalize_relative_path_blocks_traversal(monkeypatch, temp_dir: Path):
    vault = temp_dir / "vault"
    vault.mkdir()
    monkeypatch.setattr(config, "VAULT_PATH", vault)

    with pytest.raises(PermissionError):
        _normalize_relative_path("../secret.txt")


def test_normalize_relative_path_blocks_absolute_paths(monkeypatch, temp_dir: Path):
    vault = temp_dir / "vault"
    vault.mkdir()
    monkeypatch.setattr(config, "VAULT_PATH", vault)

    outside = temp_dir / "outside.md"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        _normalize_relative_path(str(outside.resolve()))
