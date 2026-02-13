"""
File routes: /file, /tags, /backlinks
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.api.models.files import FileResponse, TagsResponse, BacklinksResponse
from app.services.file_service import file_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Files"])


@router.get("/file", response_model=FileResponse)
async def get_file(
    path: str = Query(..., description="Relative path to the file in the vault"),
):
    """
    Get full content of a specific file with metadata.

    Path should be relative to the vault root.
    Returns the file content along with tags, timestamps, and frontmatter.
    If file is not in database, reads directly from filesystem and triggers indexing.
    """
    try:
        return file_service.get_file(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except Exception as e:
        logger.error("Error reading file: %s", e)
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")


@router.get("/tags", response_model=TagsResponse, tags=["Metadata"])
async def list_tags():
    """
    List all tags in the vault with usage counts.

    Returns tags sorted by usage count (descending), then alphabetically.
    Useful for exploring the knowledge graph and tag-based navigation.
    """
    return file_service.list_tags()


@router.get("/backlinks", response_model=BacklinksResponse, tags=["Metadata"])
async def get_backlinks(
    path: str = Query(..., description="Relative path or filename to find backlinks for"),
):
    """
    Find all files that link to a specific file.

    Searches for wikilinks ([[filename]]) and markdown links that reference
    the target file. Useful for exploring connections in the knowledge graph.
    """
    return file_service.get_backlinks(path)
