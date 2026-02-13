"""
Sync routes: /sync, /sync/stats, /sync/file
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.api.models.sync import SyncRequest, SyncResponse, PostgresStatsResponse
from app.services.sync_api_service import sync_api_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Sync"])


@router.post("/sync", response_model=SyncResponse)
async def sync_to_postgres(request: SyncRequest):
    """
    Synchronize SQLite data to PostgreSQL.

    This enables a Next.js frontend to access the Second Brain data via Prisma.

    Modes:
    - incremental: Only sync files changed since last sync (faster)
    - full: Clear and rebuild all PostgreSQL data (slower but ensures consistency)
    """
    try:
        return await sync_api_service.sync(mode=request.mode)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Sync error: %s", e)
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@router.get("/sync/stats", response_model=PostgresStatsResponse)
async def postgres_stats():
    """
    Get PostgreSQL database statistics.

    Returns counts and last sync time for the PostgreSQL database.
    """
    try:
        return await sync_api_service.get_stats()
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("PostgreSQL stats error: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@router.post("/sync/file")
async def sync_single_file(path: str = Query(..., description="File path to sync")):
    """
    Sync a single file to PostgreSQL.

    Use this for on-demand syncing when a file is updated.
    """
    try:
        success = await sync_api_service.sync_file(path)
        if success:
            return {"status": "synced", "path": path}
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File sync error: %s", e)
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")
