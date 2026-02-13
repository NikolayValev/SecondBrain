"""
Inbox-related API models: processing, file listing.
"""

from typing import Optional

from pydantic import BaseModel


class InboxProcessRequest(BaseModel):
    """Request body for inbox processing."""
    dry_run: bool = False


class InboxFileResult(BaseModel):
    """Result for a single processed file."""
    source_path: str
    destination_path: Optional[str] = None
    action: str
    classification: str
    added_tags: list[str] = []
    error: Optional[str] = None


class InboxProcessResponse(BaseModel):
    """Response for inbox processing."""
    processed: int
    moved: int
    skipped: int
    errors: int
    duration_seconds: float
    results: list[InboxFileResult]


class InboxFileInfo(BaseModel):
    """Information about a file in the inbox."""
    name: str
    path: str
    size_bytes: int
    modified: str
    type: str = "file"


class InboxFolderInfo(BaseModel):
    """Information about a folder in the inbox."""
    name: str
    path: str
    type: str = "folder"
    files: list[InboxFileInfo] = []
    folders: list["InboxFolderInfo"] = []


class InboxContentsResponse(BaseModel):
    """Full contents of the inbox folder."""
    inbox_path: str
    total_files: int
    total_folders: int
    root_files: list[InboxFileInfo]
    folders: list[InboxFolderInfo]
