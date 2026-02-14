"""
Sync API service: PostgreSQL sync operations exposed to the API layer.

This thin wrapper exists so that route handlers don't need to touch
low-level sync internals or repeat postgres-availability checks.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from app.config import config
from app.db import db
from app.db_postgres import get_postgres_db
from app.sync_service import get_sync_service, run_sync
from app.api.models.sync import SyncResponse, PostgresStatsResponse

logger = logging.getLogger(__name__)


class SyncAPIService:
    """API-level sync operations (delegates to app.sync_service)."""

    @staticmethod
    def _require_postgres() -> None:
        if not config.POSTGRES_URL:
            raise EnvironmentError(
                "PostgreSQL not configured. Set DATABASE_URL or POSTGRES_URL environment variable."
            )

    async def sync(self, mode: str = "incremental") -> SyncResponse:
        """Run a full or incremental sync to PostgreSQL."""
        self._require_postgres()

        if mode not in ("full", "incremental"):
            raise ValueError("Mode must be 'full' or 'incremental'")

        stats = await run_sync(mode=mode)

        return SyncResponse(
            files_added=stats["files_added"],
            files_updated=stats["files_updated"],
            files_deleted=stats["files_deleted"],
            sections=stats.get("sections", 0),
            tags=stats.get("tags", 0),
            links=stats.get("links", 0),
            chunks=stats.get("chunks", 0),
            embeddings=stats.get("embeddings", 0),
            conversations=stats.get("conversations", 0),
            messages=stats.get("messages", 0),
            errors=stats.get("errors", []),
            status="completed" if not stats.get("errors") else "completed_with_errors",
        )

    async def get_stats(self) -> PostgresStatsResponse:
        """Return PostgreSQL database statistics."""
        self._require_postgres()
        pg_db = get_postgres_db()
        stats = await pg_db.get_stats()
        return PostgresStatsResponse(
            file_count=stats["file_count"],
            section_count=stats["section_count"],
            tag_count=stats["tag_count"],
            link_count=stats["link_count"],
            chunk_count=stats["chunk_count"],
            embedding_count=stats["embedding_count"],
            embedding_vector_count=stats.get("embedding_vector_count", 0),
            conversation_count=stats.get("conversation_count", 0),
            message_count=stats.get("message_count", 0),
            last_sync=stats.get("last_sync"),
        )

    async def sync_file(self, path: str) -> bool:
        """Sync a single file. Returns True on success."""
        self._require_postgres()
        svc = get_sync_service()
        return await svc.sync_file(path)

    async def sync_conversations(self) -> dict:
        """Sync conversations & messages only (lightweight)."""
        self._require_postgres()
        svc = get_sync_service()
        stats = {"conversations": 0, "messages": 0, "errors": []}
        try:
            await svc._sync_conversations(stats)
        except Exception as e:
            stats["errors"].append({"entity": "conversations", "error": str(e)})
        stats["status"] = "completed" if not stats["errors"] else "completed_with_errors"
        return stats

    async def get_changes_since(self, since: Optional[str] = None) -> dict:
        """
        Return counts of SQLite entities modified since *since*.

        The frontend can poll this to decide when to trigger a full/incremental sync.
        """
        # Parse the optional ISO timestamp
        since_ts: Optional[str] = None
        if since:
            try:
                dt = datetime.fromisoformat(since)
                since_ts = dt.isoformat()
            except ValueError:
                since_ts = since  # best-effort

        changed_files = 0
        changed_chunks = 0
        changed_embeddings = 0
        changed_conversations = 0

        all_files = db.get_all_files()
        for f in all_files:
            if since_ts is None or f.get("mtime", 0) > datetime.fromisoformat(since_ts).timestamp() if since_ts else True:
                changed_files += 1

        # Simple approach: count totals from SQLite
        with db.cursor() as cur:
            if since_ts:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE id > 0")
                changed_chunks = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM embeddings WHERE id > 0")
                changed_embeddings = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM conversations WHERE updated_at > ?",
                    (since_ts,),
                )
                changed_conversations = cur.fetchone()[0]
            else:
                cur.execute("SELECT COUNT(*) FROM chunks")
                changed_chunks = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM embeddings")
                changed_embeddings = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM conversations")
                changed_conversations = cur.fetchone()[0]

        return {
            "since": since,
            "files": changed_files,
            "chunks": changed_chunks,
            "embeddings": changed_embeddings,
            "conversations": changed_conversations,
            "has_changes": (changed_files + changed_chunks + changed_embeddings + changed_conversations) > 0,
        }


# Singleton
sync_api_service = SyncAPIService()
