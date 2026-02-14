"""
RAG routes: /ask, /embeddings/*
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.models.rag import AskRequest, AskResponse, EmbeddingStatsResponse
from app.services.rag_service import rag_api_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question using RAG (Retrieval-Augmented Generation).

    Searches the knowledge base for relevant content and uses an LLM
    to generate an answer based on the retrieved context.

    Supports multiple providers, models, and RAG techniques.

    Set ``stream: true`` to receive the answer as Server-Sent Events.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.stream:
        return StreamingResponse(
            rag_api_service.ask_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        return await rag_api_service.ask(request)
    except ValueError as e:
        logger.error("RAG configuration error: %s", e)
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "code": "RAG_ERROR"},
        )
    except Exception as e:
        logger.error("RAG error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate answer: {e}", "code": "GENERATION_ERROR"},
        )


@router.get("/embeddings/stats", response_model=EmbeddingStatsResponse)
async def get_embedding_stats():
    """
    Get embedding statistics.
    Shows how many chunks have been embedded.
    """
    return rag_api_service.get_embedding_stats()


@router.post("/embeddings/generate")
async def generate_embeddings(
    limit: int = Query(100, ge=1, le=1000, description="Maximum chunks to process"),
):
    """
    Generate embeddings for pending chunks.

    Processes chunks that don't have embeddings yet.
    Run this after indexing to enable RAG functionality.
    """
    try:
        return await rag_api_service.generate_embeddings(limit=limit)
    except ValueError as e:
        logger.error("Embedding configuration error: %s", e)
        raise HTTPException(status_code=503, detail=f"LLM not configured: {e}")
    except Exception as e:
        logger.error("Embedding generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
