"""
Tests for /sync/changes timestamp semantics.
"""

import struct
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import register_middleware
from app.api.routes.sync import router as sync_router
from app.config import config
from app.services.sync_api_service import SyncAPIService, sync_api_service
import app.services.sync_api_service as sync_api_module


def _utc_naive_iso(dt: datetime) -> str:
    dt_utc = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt_utc.replace(tzinfo=None).isoformat()


def _seed_sync_entities(temp_db):
    old_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    new_dt = datetime(2026, 1, 3, 0, 0, 0, tzinfo=timezone.utc)

    old_file_id = temp_db.upsert_file(
        path="old.md",
        mtime=old_dt.timestamp(),
        title="Old",
        content="old",
    )
    new_file_id = temp_db.upsert_file(
        path="new.md",
        mtime=new_dt.timestamp(),
        title="New",
        content="new",
    )

    old_chunk_id = temp_db.add_chunk(file_id=old_file_id, chunk_index=0, content="old chunk", token_count=2)
    new_chunk_id = temp_db.add_chunk(file_id=new_file_id, chunk_index=0, content="new chunk", token_count=2)

    emb_old = struct.pack("3f", 0.1, 0.2, 0.3)
    emb_new = struct.pack("3f", 0.4, 0.5, 0.6)
    temp_db.add_embedding(chunk_id=old_chunk_id, embedding=emb_old, model="test", dimensions=3)
    temp_db.add_embedding(chunk_id=new_chunk_id, embedding=emb_new, model="test", dimensions=3)

    old_conversation_id = temp_db.create_conversation(title="Old convo")
    new_conversation_id = temp_db.create_conversation(title="New convo")

    with temp_db.cursor() as cur:
        cur.execute(
            "UPDATE embeddings SET created_at = ? WHERE chunk_id = ?",
            (_utc_naive_iso(old_dt), old_chunk_id),
        )
        cur.execute(
            "UPDATE embeddings SET created_at = ? WHERE chunk_id = ?",
            (_utc_naive_iso(new_dt), new_chunk_id),
        )
        cur.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utc_naive_iso(old_dt), old_conversation_id),
        )
        cur.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utc_naive_iso(new_dt), new_conversation_id),
        )


@pytest.fixture
def sync_service_with_temp_db(monkeypatch, temp_db):
    monkeypatch.setattr(sync_api_module, "db", temp_db)
    return SyncAPIService(), temp_db


@pytest.mark.asyncio
async def test_get_changes_since_filters_all_entities(sync_service_with_temp_db):
    service, temp_db = sync_service_with_temp_db
    _seed_sync_entities(temp_db)

    result = await service.get_changes_since("2026-01-02T00:00:00Z")

    assert result["files"] == 1
    assert result["chunks"] == 1
    assert result["embeddings"] == 1
    assert result["conversations"] == 1
    assert result["has_changes"] is True


@pytest.mark.asyncio
async def test_get_changes_since_without_timestamp_returns_totals(sync_service_with_temp_db):
    service, temp_db = sync_service_with_temp_db
    _seed_sync_entities(temp_db)

    result = await service.get_changes_since()

    assert result["files"] == 2
    assert result["chunks"] == 2
    assert result["embeddings"] == 2
    assert result["conversations"] == 2


@pytest.mark.asyncio
async def test_get_changes_since_rejects_invalid_timestamp(sync_service_with_temp_db):
    service, _ = sync_service_with_temp_db

    with pytest.raises(ValueError, match="Invalid 'since' timestamp"):
        await service.get_changes_since("not-a-timestamp")


def test_sync_changes_route_returns_400_for_invalid_since(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)

    async def _raise_value_error(since=None):
        raise ValueError("Invalid 'since' timestamp. Use ISO-8601 format.")

    monkeypatch.setattr(sync_api_service, "get_changes_since", _raise_value_error)

    app = FastAPI()
    register_middleware(app)
    app.include_router(sync_router)

    client = TestClient(app)
    response = client.get("/sync/changes?since=bad-input", headers={"X-API-Key": "x" * 32})

    assert response.status_code == 400
    assert "Invalid 'since' timestamp" in response.json()["detail"]
