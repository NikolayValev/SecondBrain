"""
Indexing routes: /reindex, /index, /index/status
"""

import asyncio
import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.api.models.indexing import IndexRequest, IndexResponse, IndexStatusResponse
from app.config import config
from app.services.indexing_service import indexing_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Indexing"])


def _parse_interval_ms(websocket: WebSocket) -> int:
    """Parse and clamp websocket polling interval query parameter."""
    raw = websocket.query_params.get("interval_ms")
    if not raw:
        return 1000
    try:
        value = int(raw)
    except ValueError:
        return 1000
    return max(100, min(10000, value))


async def _authorize_websocket(websocket: WebSocket) -> bool:
    """Apply API key validation for websocket endpoints."""
    if not config.BRAIN_API_KEY:
        if config.REQUIRE_API_KEY:
            await websocket.close(code=4401, reason="API key required")
            return False
        return True

    api_key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    if not api_key:
        await websocket.close(code=4401, reason="Missing API key")
        return False

    if not hmac.compare_digest(api_key, config.BRAIN_API_KEY):
        await websocket.close(code=4403, reason="Invalid API key")
        return False

    return True


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
        raise HTTPException(status_code=500, detail="Reindex failed")


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


@router.websocket("/ws/index/status")
async def stream_index_status(websocket: WebSocket):
    """
    Stream indexing status updates over WebSocket.

    Query params:
    - interval_ms: polling interval in milliseconds (default 1000, clamped 100-10000)

    Auth:
    - When API key auth is enabled, send `X-API-Key` header or `api_key` query param.
    """
    authorized = await _authorize_websocket(websocket)
    if not authorized:
        return

    await websocket.accept()
    interval_seconds = _parse_interval_ms(websocket) / 1000.0

    try:
        while True:
            status = indexing_service.get_status()
            await websocket.send_json(status.model_dump())
            await asyncio.sleep(interval_seconds)
    except WebSocketDisconnect:
        logger.debug("Index status websocket disconnected")
    except Exception as e:
        logger.error("Index status websocket error: %s", e)
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass
