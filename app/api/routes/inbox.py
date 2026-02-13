"""
Inbox routes: /inbox/process, /inbox/files, /inbox/contents
"""

import logging

from fastapi import APIRouter, HTTPException

from app.api.models.inbox import (
    InboxProcessRequest,
    InboxProcessResponse,
    InboxContentsResponse,
)
from app.services.inbox_service import inbox_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inbox"])


@router.post("/inbox/process", response_model=InboxProcessResponse)
async def process_inbox(request: InboxProcessRequest):
    """
    Process documents in the inbox folder.

    Classifies, tags, and moves documents from 00_Inbox to appropriate folders.
    Use dry_run=True to preview changes without moving files.
    """
    try:
        return await inbox_service.process(request)
    except Exception as e:
        logger.error("Inbox processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inbox processing failed: {e}")


@router.get("/inbox/files")
async def list_inbox_files():
    """
    List files currently in the inbox.

    Returns files that would be processed on next inbox run.
    """
    return inbox_service.list_files()


@router.get("/inbox/contents", response_model=InboxContentsResponse)
async def get_inbox_contents():
    """
    Get the full contents of the inbox folder including all subfolders.

    Returns a hierarchical view of all files and folders in 00_Inbox.
    """
    try:
        return inbox_service.get_contents()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
