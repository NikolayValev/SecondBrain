"""
Indexing service: full/incremental scans and background job tracking.
"""

import logging
import uuid
from datetime import datetime

from app.config import config
from app.db import db
from app.indexer import indexer
from app.api.models.indexing import IndexRequest, IndexResponse, IndexStatusResponse

logger = logging.getLogger(__name__)

# In-memory job store (survives only within a single process)
_indexing_jobs: dict[str, dict] = {}


class IndexingService:
    """Manages indexing operations and background jobs."""

    def reindex(self, full: bool = False) -> dict:
        """
        Run an immediate (synchronous) reindex.

        Returns:
            dict with status, indexed count, errors, type.
        """
        if full:
            indexed, errors = indexer.full_scan()
        else:
            indexed, errors = indexer.incremental_scan()

        return {
            "status": "completed",
            "indexed": indexed,
            "errors": errors,
            "type": "full" if full else "incremental",
        }

    def start_background_index(self, request: IndexRequest) -> tuple[str, int]:
        """
        Create a background indexing job.

        Returns:
            (job_id, documents_queued)
        """
        job_id = str(uuid.uuid4())

        docs_queued = len(request.paths) if request.paths else sum(1 for _ in config.VAULT_PATH.rglob("*.md"))

        _indexing_jobs[job_id] = {
            "status": "started",
            "progress": 0,
            "documents_processed": 0,
            "documents_total": docs_queued,
            "started_at": datetime.now().isoformat(),
        }

        return job_id, docs_queued

    async def run_indexing_job(self, job_id: str, request: IndexRequest) -> None:
        """Execute the indexing work (called from a BackgroundTask)."""
        try:
            _indexing_jobs[job_id]["status"] = "indexing"

            if request.paths:
                total = len(request.paths)
                processed = 0
                for path in request.paths:
                    try:
                        full_path = config.VAULT_PATH / path
                        if full_path.exists():
                            indexer.index_file(str(full_path), force=request.force)
                            processed += 1
                    except Exception as e:
                        logger.error("Failed to index %s: %s", path, e)

                    _indexing_jobs[job_id]["documents_processed"] = processed
                    _indexing_jobs[job_id]["progress"] = processed / total if total > 0 else 1

                _indexing_jobs[job_id]["documents_total"] = total
            else:
                if request.force:
                    indexed, errors = indexer.full_scan()
                else:
                    indexed, errors = indexer.incremental_scan()

                _indexing_jobs[job_id]["documents_processed"] = indexed
                _indexing_jobs[job_id]["documents_total"] = indexed + errors

            _indexing_jobs[job_id]["status"] = "completed"
            _indexing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            logger.error("Indexing job %s failed: %s", job_id, e)
            _indexing_jobs[job_id]["status"] = "error"
            _indexing_jobs[job_id]["error"] = str(e)

    def get_status(self) -> IndexStatusResponse:
        """Return the current indexing status."""
        stats = db.get_stats()
        embedding_stats = db.get_embedding_stats()

        current_job = None
        for job_id, job in _indexing_jobs.items():
            if job["status"] in ("started", "indexing"):
                current_job = {
                    "job_id": job_id,
                    "progress": job.get("progress", 0),
                    "documents_processed": job.get("documents_processed", 0),
                    "documents_total": job.get("documents_total", 0),
                }
                break

        return IndexStatusResponse(
            status="indexing" if current_job else "idle",
            documents_indexed=stats["file_count"],
            documents_pending=embedding_stats.get("pending_chunks", 0),
            last_indexed_at=stats.get("last_indexed"),
            current_job=current_job,
        )


# Singleton
indexing_service = IndexingService()
