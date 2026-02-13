"""
Indexing-related API models.
"""

from typing import Optional

from pydantic import BaseModel


class IndexRequest(BaseModel):
    """Request for indexing."""
    paths: Optional[list[str]] = None
    force: bool = False


class IndexResponse(BaseModel):
    """Response for index operation."""
    status: str
    job_id: str
    documents_queued: int


class IndexStatusResponse(BaseModel):
    """Response for index status."""
    status: str  # idle, indexing, error
    documents_indexed: int
    documents_pending: int
    last_indexed_at: Optional[str] = None
    current_job: Optional[dict] = None
