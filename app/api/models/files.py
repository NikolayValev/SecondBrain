"""
File-related API models: files, tags, backlinks.
"""

from typing import Optional

from pydantic import BaseModel


class FileResponse(BaseModel):
    """File content response with metadata."""
    path: str
    title: str
    content: str
    tags: list[str] = []
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    frontmatter: dict = {}


class BacklinkItem(BaseModel):
    """A single backlink item."""
    path: str
    title: str


class BacklinksResponse(BaseModel):
    """Response for backlinks endpoint."""
    target: str
    backlinks: list[BacklinkItem]
    count: int


class TagItem(BaseModel):
    """A single tag with usage count."""
    name: str
    file_count: int


class TagsResponse(BaseModel):
    """Response for tags endpoint."""
    tags: list[TagItem]
    count: int
