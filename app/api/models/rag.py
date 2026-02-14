"""
RAG-related API models: ask, sources, embeddings.
"""

from typing import Optional

from pydantic import BaseModel

from app.config import Config


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""
    question: str
    conversation_id: Optional[str] = None
    provider: str = Config.LLM_PROVIDER
    model: Optional[str] = None
    rag_technique: str = "hybrid"
    include_sources: bool = True
    stream: bool = False


class Source(BaseModel):
    """Source information for responses."""
    path: str
    title: str
    snippet: str
    score: float


class TokenUsage(BaseModel):
    """Token usage information."""
    prompt: int = 0
    completion: int = 0
    total: int = 0


class SourceInfo(BaseModel):
    """Source information for RAG response."""
    file_path: str
    file_title: str
    section: Optional[str] = None
    similarity: float


class AskResponse(BaseModel):
    """Response from /ask endpoint."""
    answer: str
    sources: list[Source]
    conversation_id: Optional[str] = None
    model_used: str
    tokens_used: Optional[TokenUsage] = None


class EmbeddingStatsResponse(BaseModel):
    """Embedding statistics response."""
    chunk_count: int
    embedding_count: int
    files_with_embeddings: int
    pending_chunks: int
