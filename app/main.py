"""
Second Brain Daemon - FastAPI Application
A local daemon that indexes an Obsidian vault and exposes a search API.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import config
from app.db import db
from app.indexer import indexer
from app.watcher import watcher

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# Response models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vault_path: str
    watcher_running: bool


class StatsResponse(BaseModel):
    """Database statistics response."""
    file_count: int
    section_count: int
    tag_count: int
    link_count: int
    last_indexed: Optional[str]


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


class FileResponse(BaseModel):
    """File content response."""
    path: str
    title: str
    content: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Second Brain daemon...")
    
    try:
        # Validate configuration
        config.validate()
        logger.info(f"Vault path: {config.VAULT_PATH}")
        
        # Initialize database
        db.initialize()
        
        # Perform initial scan
        stats = db.get_stats()
        if stats["file_count"] == 0:
            logger.info("First run detected - performing full vault scan")
            indexed, errors = indexer.full_scan()
            logger.info(f"Initial scan complete: {indexed} files indexed, {errors} errors")
        else:
            logger.info("Existing index found - performing incremental scan")
            indexed, errors = indexer.incremental_scan()
            if indexed > 0:
                logger.info(f"Incremental scan: {indexed} files updated, {errors} errors")
        
        # Start file watcher
        watcher.start()
        
        logger.info("Second Brain daemon started successfully")
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Second Brain daemon...")
    
    watcher.stop()
    db.close()
    
    logger.info("Second Brain daemon stopped")


# Create FastAPI app
app = FastAPI(
    title="Second Brain API",
    description="Local API for searching and accessing an indexed Obsidian vault",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the daemon and vault path.
    """
    return HealthResponse(
        status="healthy",
        vault_path=str(config.VAULT_PATH),
        watcher_running=watcher.is_running
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get database statistics.
    Returns counts of files, sections, tags, links, and last indexed time.
    """
    stats = db.get_stats()
    return StatsResponse(
        file_count=stats["file_count"],
        section_count=stats["section_count"],
        tag_count=stats["tag_count"],
        link_count=stats["link_count"],
        last_indexed=stats["last_indexed"]
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return")
):
    """
    Full-text search across the vault.
    Uses FTS5 for keyword matching with BM25 ranking.
    Returns snippets with highlighted matches.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = db.search(q, limit=limit)
        
        return SearchResponse(
            query=q,
            results=[
                SearchResult(
                    file_path=r["file_path"],
                    title=r["title"],
                    heading=r["heading"],
                    snippet=r["snippet"],
                    rank=r["rank"]
                )
                for r in results
            ],
            count=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/file", response_model=FileResponse, tags=["Files"])
async def get_file(
    path: str = Query(..., description="Relative path to the file in the vault")
):
    """
    Get full content of a specific file.
    Path should be relative to the vault root.
    """
    file_record = db.get_file_by_path(path)
    
    if not file_record:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}"
        )
    
    return FileResponse(
        path=file_record["path"],
        title=file_record["title"],
        content=file_record["content"]
    )


@app.get("/tags", tags=["Metadata"])
async def list_tags():
    """
    List all tags in the vault.
    Useful for exploring the knowledge graph.
    """
    with db.cursor() as cur:
        cur.execute("""
            SELECT t.name, COUNT(ft.file_id) as file_count
            FROM tags t
            LEFT JOIN file_tags ft ON t.id = ft.tag_id
            GROUP BY t.id
            ORDER BY file_count DESC, t.name
        """)
        tags = [{"name": row["name"], "file_count": row["file_count"]} for row in cur.fetchall()]
    
    return {"tags": tags, "count": len(tags)}


@app.get("/backlinks", tags=["Metadata"])
async def get_backlinks(
    path: str = Query(..., description="Relative path or filename to find backlinks for")
):
    """
    Find all files that link to a specific file.
    Useful for exploring connections in the knowledge graph.
    """
    # Normalize path for matching
    target_name = Path(path).stem
    
    with db.cursor() as cur:
        # Find links that match the path or filename
        cur.execute("""
            SELECT DISTINCT f.path, f.title
            FROM links l
            JOIN files f ON l.from_file_id = f.id
            WHERE l.to_path = ? OR l.to_path = ? OR l.to_path LIKE ?
        """, (path, target_name, f"%{target_name}%"))
        
        backlinks = [{"path": row["path"], "title": row["title"]} for row in cur.fetchall()]
    
    return {"target": path, "backlinks": backlinks, "count": len(backlinks)}


@app.post("/reindex", tags=["System"])
async def trigger_reindex(full: bool = Query(False, description="Perform full rescan")):
    """
    Manually trigger a reindex of the vault.
    Use full=true to force a complete rescan.
    """
    try:
        if full:
            indexed, errors = indexer.full_scan()
        else:
            indexed, errors = indexer.incremental_scan()
        
        return {
            "status": "completed",
            "indexed": indexed,
            "errors": errors,
            "type": "full" if full else "incremental"
        }
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid request", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Entry point for running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
