"""
Search-related API models: full-text and semantic search.
"""

from typing import Optional

from pydantic import BaseModel


class SearchResult(BaseModel):
    """Single search result."""
    file_path: str
    title: str
    heading: str
    snippet: str
    rank: float


class SearchResponse(BaseModel):
    """Search results response."""
    query: str
    results: list[SearchResult]
    count: int


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    query: str
    limit: int = 10
    rag_technique: str = "basic"


class SemanticSearchResult(BaseModel):
    """A single semantic search result."""
    path: str
    title: str
    snippet: str
    score: float
    metadata: dict = {}


class SemanticSearchResponse(BaseModel):
    """Response for semantic search."""
    results: list[SemanticSearchResult]
    query_embedding_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None
