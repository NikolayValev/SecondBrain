"""
Indexing routes: /reindex, /index, /index/status
"""

import logging

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks

from app.api.models.indexing import IndexRequest, IndexResponse, IndexStatusResponse
from app.services.indexing_service import indexing_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Indexing"])


@router.post("/reindex")
async def trigger_reindex(full: bool = Query(False, description="Perform full rescan")):
    """
    Manually trigger a reindex of the vault.
    Use full=true to force a complete rescan.
    """
    try:
        return indexing_service.reindex(full=full)
    except Exception as e:
        logger.error("Reindex error: %s", e)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


@router.post("/index", response_model=IndexResponse)
async def trigger_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Trigger document indexing for the knowledge base.

    Runs indexing in the background and returns a job ID for tracking.
    """
    job_id, docs_queued = indexing_service.start_background_index(request)

    background_tasks.add_task(indexing_service.run_indexing_job, job_id, request)

    return IndexResponse(
        status="started",
        job_id=job_id,
        documents_queued=docs_queued,
    )


@router.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """
    Get the current indexing status.
    """
    return indexing_service.get_status()
