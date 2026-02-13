"""
System routes: /health, /stats, /config
"""

from fastapi import APIRouter

from app.api.models.system import HealthResponse, StatsResponse, ConfigResponse
from app.services.system_service import system_service

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Enhanced health check endpoint.
    Returns the status of the daemon, providers, and vector store.
    """
    return await system_service.get_health()


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database statistics.
    Returns counts of files, sections, tags, links, and last indexed time.
    """
    return system_service.get_stats()


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get available configuration options.
    Returns available providers, models, and RAG techniques.
    """
    return await system_service.get_config()
