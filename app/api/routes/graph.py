"""
Graph routes: /graph/links
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.api.models.graph import GraphResponse
from app.services.graph_service import graph_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Graph"])


@router.get("/graph/links", response_model=GraphResponse)
async def get_links_graph(
    max_edges: int = Query(2000, ge=1, le=20000, description="Maximum edges to include"),
    include_dangling: bool = Query(True, description="Include unresolved link targets"),
):
    """
    Return a note-link graph suitable for frontend visualization.

    - Nodes represent indexed notes and optional unresolved targets.
    - Directed edges represent outbound links found in note content.
    """
    try:
        return graph_service.get_links_graph(
            max_edges=max_edges,
            include_dangling=include_dangling,
        )
    except Exception as e:
        logger.error("Graph generation error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to build graph")
