"""
Sync API service: PostgreSQL sync operations exposed to the API layer.

This thin wrapper exists so that route handlers don't need to touch
low-level sync internals or repeat postgres-availability checks.
"""

import logging

from app.config import config
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
            last_sync=stats.get("last_sync"),
        )

    async def sync_file(self, path: str) -> bool:
        """Sync a single file. Returns True on success."""
        self._require_postgres()
        svc = get_sync_service()
        return await svc.sync_file(path)


# Singleton
sync_api_service = SyncAPIService()
