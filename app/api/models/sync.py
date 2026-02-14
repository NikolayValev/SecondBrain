"""
Sync-related API models: PostgreSQL sync.
"""

from typing import Optional

from pydantic import BaseModel


class SyncRequest(BaseModel):
    """Request for PostgreSQL sync."""
    mode: str = "incremental"  # "full" or "incremental"


class SyncResponse(BaseModel):
    """Response from PostgreSQL sync."""
    files_added: int
    files_updated: int
    files_deleted: int
    sections: int
    tags: int
    links: int
    chunks: int
    embeddings: int
    conversations: int = 0
    messages: int = 0
    errors: list[dict]
    status: str


class PostgresStatsResponse(BaseModel):
    """PostgreSQL database statistics."""
    file_count: int
    section_count: int
    tag_count: int
    link_count: int
    chunk_count: int
    embedding_count: int
    embedding_vector_count: int = 0
    conversation_count: int = 0
    message_count: int = 0
    last_sync: Optional[str]
