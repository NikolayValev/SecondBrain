"""
Tests for websocket indexing status updates.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.middleware import register_middleware
from app.api.models.indexing import IndexStatusResponse
from app.api.routes.indexing import router as indexing_router
from app.config import config
import app.api.routes.indexing as indexing_module


def _build_app(monkeypatch) -> FastAPI:
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)

    app = FastAPI()
    register_middleware(app)
    app.include_router(indexing_router)
    return app


def test_index_status_websocket_streams_status(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)

    calls = {"count": 0}

    def fake_get_status():
        calls["count"] += 1
        return IndexStatusResponse(
            status="indexing",
            documents_indexed=10,
            documents_pending=2,
            last_indexed_at="2026-02-25T20:00:00",
            current_job={
                "job_id": "job-1",
                "progress": 0.4,
                "documents_processed": 4,
                "documents_total": 10,
            },
        )

    monkeypatch.setattr(indexing_module.indexing_service, "get_status", fake_get_status)

    with client.websocket_connect(
        "/ws/index/status?interval_ms=100",
        headers={"X-API-Key": "x" * 32},
    ) as ws:
        message = ws.receive_json()
        assert message["status"] == "indexing"
        assert message["documents_indexed"] == 10
        assert message["current_job"]["job_id"] == "job-1"

    assert calls["count"] >= 1


def test_index_status_websocket_requires_api_key(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/index/status") as ws:
            ws.receive_json()

    assert exc.value.code == 4401
