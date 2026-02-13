"""
Search routes: GET /search (full-text), POST /search (semantic)
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.api.models.search import (
    SearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.services.search_service import search_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
):
    """
    Full-text search across the vault.
    Uses FTS5 for keyword matching with BM25 ranking.
    Returns snippets with highlighted matches.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        return search_service.fulltext_search(q, limit=limit)
    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search across the vault (without LLM generation).

    Uses embeddings for semantic similarity matching.
    Supports different RAG techniques for retrieval.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        return await search_service.semantic_search(
            query=request.query,
            limit=request.limit,
            rag_technique=request.rag_technique,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Semantic search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
