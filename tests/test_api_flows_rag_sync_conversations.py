"""
API flow tests for RAG, Sync, and Conversations routes.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import register_middleware
from app.api.models.rag import AskResponse
from app.api.models.sync import PostgresStatsResponse, SyncResponse
from app.api.routes.conversations import router as conversations_router
from app.api.routes.rag import router as rag_router
from app.api.routes.sync import router as sync_router
from app.config import config
import app.services.conversation_service as conversation_module
from app.services.rag_service import rag_api_service
from app.services.sync_api_service import sync_api_service


def _build_app(monkeypatch, *, routers: list) -> FastAPI:
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
    for router in routers:
        app.include_router(router)
    return app


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "x" * 32}


def test_ask_route_non_stream_flow(monkeypatch):
    app = _build_app(monkeypatch, routers=[rag_router])
    client = TestClient(app)

    async def fake_ask(request):
        return AskResponse(
            answer="Mock answer",
            sources=[],
            conversation_id="7",
            model_used="mock-model",
        )

    monkeypatch.setattr(rag_api_service, "ask", fake_ask)

    response = client.post(
        "/ask",
        json={"question": "What is this?", "provider": "openai", "stream": False},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mock answer"
    assert data["conversation_id"] == "7"


def test_ask_route_stream_flow(monkeypatch):
    app = _build_app(monkeypatch, routers=[rag_router])
    client = TestClient(app)

    async def fake_stream(request):
        yield 'data: {"type":"token","content":"Hello"}\n\n'
        yield 'data: {"type":"done","conversation_id":"7","model_used":"mock-model"}\n\n'

    monkeypatch.setattr(rag_api_service, "ask_stream", fake_stream)

    response = client.post(
        "/ask",
        json={"question": "Stream?", "provider": "openai", "stream": True},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert '"type":"done"' in response.text


def test_sync_routes_flow(monkeypatch):
    app = _build_app(monkeypatch, routers=[sync_router])
    client = TestClient(app)

    async def fake_sync(mode: str = "incremental"):
        return SyncResponse(
            files_added=1,
            files_updated=2,
            files_deleted=0,
            sections=3,
            tags=4,
            links=5,
            chunks=6,
            embeddings=7,
            conversations=8,
            messages=9,
            errors=[],
            status="completed",
        )

    async def fake_get_stats():
        return PostgresStatsResponse(
            file_count=1,
            section_count=2,
            tag_count=3,
            link_count=4,
            chunk_count=5,
            embedding_count=6,
            embedding_vector_count=7,
            conversation_count=8,
            message_count=9,
            last_sync="2026-02-25T00:00:00",
        )

    async def fake_sync_file(path: str):
        return path == "exists.md"

    async def fake_sync_conversations():
        return {"status": "completed", "conversations": 1, "messages": 2, "errors": []}

    seen_since: dict[str, str] = {}

    async def fake_get_changes_since(since: str | None = None):
        seen_since["value"] = since or ""
        return {
            "since": since,
            "files": 1,
            "chunks": 2,
            "embeddings": 3,
            "conversations": 4,
            "has_changes": True,
        }

    monkeypatch.setattr(sync_api_service, "sync", fake_sync)
    monkeypatch.setattr(sync_api_service, "get_stats", fake_get_stats)
    monkeypatch.setattr(sync_api_service, "sync_file", fake_sync_file)
    monkeypatch.setattr(sync_api_service, "sync_conversations", fake_sync_conversations)
    monkeypatch.setattr(sync_api_service, "get_changes_since", fake_get_changes_since)

    sync_response = client.post("/sync", json={"mode": "incremental"}, headers=_auth_headers())
    assert sync_response.status_code == 200
    assert sync_response.json()["status"] == "completed"

    stats_response = client.get("/sync/stats", headers=_auth_headers())
    assert stats_response.status_code == 200
    assert stats_response.json()["embedding_vector_count"] == 7

    file_ok = client.post("/sync/file?path=exists.md", headers=_auth_headers())
    assert file_ok.status_code == 200
    assert file_ok.json()["status"] == "synced"

    file_missing = client.post("/sync/file?path=missing.md", headers=_auth_headers())
    assert file_missing.status_code == 404

    conversations_response = client.post("/sync/conversations", headers=_auth_headers())
    assert conversations_response.status_code == 200
    assert conversations_response.json()["messages"] == 2

    changes_response = client.get(
        "/sync/changes?since=2026-02-24T00:00:00",
        headers=_auth_headers(),
    )
    assert changes_response.status_code == 200
    assert seen_since["value"] == "2026-02-24T00:00:00"


def test_conversations_route_full_flow(monkeypatch, temp_db):
    monkeypatch.setattr(config, "POSTGRES_URL", "")
    monkeypatch.setattr(conversation_module, "db", temp_db)

    app = _build_app(monkeypatch, routers=[conversations_router])
    client = TestClient(app)
    headers = _auth_headers()

    create_response = client.post(
        "/conversations",
        json={
            "session_id": "session-a",
            "title": "Test chat",
            "system_prompt": "Keep answers short and actionable.",
        },
        headers=headers,
    )
    assert create_response.status_code == 200
    conversation_id = create_response.json()["id"]

    add_response = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "Hello"},
        headers=headers,
    )
    assert add_response.status_code == 200
    assert add_response.json()["status"] == "added"

    get_response = client.get(f"/conversations/{conversation_id}", headers=headers)
    assert get_response.status_code == 200
    payload = get_response.json()
    assert payload["id"] == conversation_id
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "Keep answers short and actionable."
    assert payload["messages"][1]["content"] == "Hello"

    list_response = client.get("/conversations?session_id=session-a", headers=headers)
    assert list_response.status_code == 200
    listing = list_response.json()
    assert listing["count"] >= 1
    assert any(item["id"] == conversation_id for item in listing["conversations"])
