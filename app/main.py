"""
Second Brain Daemon - FastAPI Application
A local daemon that indexes an Obsidian vault and exposes a search API.
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import config
from app.db import db
from app.indexer import indexer
from app.watcher import watcher
from app.rag import rag_service
from app.embeddings import embedding_service
from app.inbox_processor import inbox_processor, InboxConfig, create_default_config
from app.sync_service import get_sync_service, run_sync
from app.db_postgres import get_postgres_db, close_postgres_db

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


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""
    question: str
    include_sources: bool = True


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


class SourceInfo(BaseModel):
    """Source information for RAG response."""
    file_path: str
    file_title: str
    section: Optional[str] = None
    similarity: float


class AskResponse(BaseModel):
    """Response from RAG-powered Q&A."""
    answer: str
    sources: list[SourceInfo]
    query: str


class EmbeddingStatsResponse(BaseModel):
    """Embedding statistics response."""
    chunk_count: int
    embedding_count: int
    files_with_embeddings: int
    pending_chunks: int


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
    last_sync: Optional[str]


class ConversationCreate(BaseModel):
    """Request to create a conversation."""
    session_id: Optional[str] = None
    title: Optional[str] = None


class MessageCreate(BaseModel):
    """Request to add a message."""
    role: str
    content: str
    sources: Optional[list[dict]] = None


class ConversationResponse(BaseModel):
    """Conversation with messages."""
    id: int
    session_id: Optional[str]
    title: Optional[str]
    created_at: str
    updated_at: str
    messages: list[dict]


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


# RAG Endpoints

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(request: AskRequest):
    """
    Ask a question using RAG (Retrieval-Augmented Generation).
    
    Searches the knowledge base for relevant content and uses an LLM
    to generate an answer based on the retrieved context.
    
    Requires LLM configuration (OPENAI_API_KEY, GEMINI_API_KEY, or Ollama).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = await rag_service.ask(
            question=request.question,
            include_sources=request.include_sources,
        )
        
        return AskResponse(
            answer=response.answer,
            sources=[
                SourceInfo(
                    file_path=s["file_path"],
                    file_title=s["file_title"],
                    section=s.get("section"),
                    similarity=s["similarity"],
                )
                for s in response.sources
            ],
            query=response.query,
        )
    except ValueError as e:
        logger.error(f"RAG configuration error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM not configured: {str(e)}. Set OPENAI_API_KEY, GEMINI_API_KEY, or configure Ollama."
        )
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@app.get("/embeddings/stats", response_model=EmbeddingStatsResponse, tags=["RAG"])
async def get_embedding_stats():
    """
    Get embedding statistics.
    Shows how many chunks have been embedded.
    """
    stats = db.get_embedding_stats()
    return EmbeddingStatsResponse(**stats)


@app.post("/inbox/process", response_model=InboxProcessResponse, tags=["Inbox"])
async def process_inbox(request: InboxProcessRequest):
    """
    Process documents in the inbox folder.
    
    Classifies, tags, and moves documents from 00_Inbox to appropriate folders.
    Use dry_run=True to preview changes without moving files.
    """
    try:
        # Create processor with dry_run setting
        processor_config = create_default_config()
        processor_config.dry_run = request.dry_run
        
        from app.inbox_processor import InboxProcessor
        processor = InboxProcessor(processor_config)
        
        result = await processor.process_inbox()
        
        return InboxProcessResponse(
            processed=result.processed,
            moved=result.moved,
            skipped=result.skipped,
            errors=result.errors,
            duration_seconds=result.duration_seconds,
            results=[
                InboxFileResult(
                    source_path=str(r.source_path),
                    destination_path=str(r.destination_path) if r.destination_path else None,
                    action=r.action,
                    classification=r.classification,
                    added_tags=r.added_tags,
                    error=r.error,
                )
                for r in result.results
            ]
        )
    except Exception as e:
        logger.error(f"Inbox processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Inbox processing failed: {str(e)}")


@app.get("/inbox/files", tags=["Inbox"])
async def list_inbox_files():
    """
    List files currently in the inbox.
    
    Returns files that would be processed on next inbox run.
    """
    files = inbox_processor.get_inbox_files()
    return {
        "count": len(files),
        "files": [
            {
                "name": f.name,
                "path": str(f.relative_to(config.VAULT_PATH)),
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in files
        ]
    }


# =============================================================================
# PostgreSQL Sync Endpoints
# =============================================================================

@app.post("/sync", response_model=SyncResponse, tags=["Sync"])
async def sync_to_postgres(request: SyncRequest):
    """
    Synchronize SQLite data to PostgreSQL.
    
    This enables a Next.js frontend to access the Second Brain data via Prisma.
    
    Modes:
    - incremental: Only sync files changed since last sync (faster)
    - full: Clear and rebuild all PostgreSQL data (slower but ensures consistency)
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured. Set DATABASE_URL or POSTGRES_URL environment variable."
        )
    
    try:
        if request.mode not in ("full", "incremental"):
            raise HTTPException(
                status_code=400,
                detail="Mode must be 'full' or 'incremental'"
            )
        
        stats = await run_sync(mode=request.mode)
        
        return SyncResponse(
            files_added=stats['files_added'],
            files_updated=stats['files_updated'],
            files_deleted=stats['files_deleted'],
            sections=stats.get('sections', 0),
            tags=stats.get('tags', 0),
            links=stats.get('links', 0),
            chunks=stats.get('chunks', 0),
            embeddings=stats.get('embeddings', 0),
            errors=stats.get('errors', []),
            status='completed' if not stats.get('errors') else 'completed_with_errors'
        )
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@app.get("/sync/stats", response_model=PostgresStatsResponse, tags=["Sync"])
async def postgres_stats():
    """
    Get PostgreSQL database statistics.
    
    Returns counts and last sync time for the PostgreSQL database.
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured. Set DATABASE_URL or POSTGRES_URL environment variable."
        )
    
    try:
        pg_db = get_postgres_db()
        stats = await pg_db.get_stats()
        
        return PostgresStatsResponse(
            file_count=stats['file_count'],
            section_count=stats['section_count'],
            tag_count=stats['tag_count'],
            link_count=stats['link_count'],
            chunk_count=stats['chunk_count'],
            embedding_count=stats['embedding_count'],
            last_sync=stats.get('last_sync')
        )
    except Exception as e:
        logger.error(f"PostgreSQL stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/sync/file", tags=["Sync"])
async def sync_single_file(path: str = Query(..., description="File path to sync")):
    """
    Sync a single file to PostgreSQL.
    
    Use this for on-demand syncing when a file is updated.
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured."
        )
    
    try:
        sync_service = get_sync_service()
        success = await sync_service.sync_file(path)
        
        if success:
            return {"status": "synced", "path": path}
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except Exception as e:
        logger.error(f"File sync error: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


# =============================================================================
# Conversation Endpoints (for Next.js chat interface)
# =============================================================================

@app.post("/conversations", tags=["Conversations"])
async def create_conversation(request: ConversationCreate):
    """
    Create a new conversation.
    
    Used by Next.js frontend to track chat sessions.
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured."
        )
    
    try:
        pg_db = get_postgres_db()
        conv_id = await pg_db.create_conversation(
            session_id=request.session_id,
            title=request.title
        )
        
        return {"id": conv_id, "status": "created"}
    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(conversation_id: int):
    """
    Get a conversation with all messages.
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured."
        )
    
    try:
        pg_db = get_postgres_db()
        conv = await pg_db.get_conversation(conversation_id)
        
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conv
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/{conversation_id}/messages", tags=["Conversations"])
async def add_message(conversation_id: int, request: MessageCreate):
    """
    Add a message to a conversation.
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured."
        )
    
    try:
        pg_db = get_postgres_db()
        message_id = await pg_db.add_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            sources=request.sources
        )
        
        return {"id": message_id, "status": "added"}
    except Exception as e:
        logger.error(f"Add message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations", tags=["Conversations"])
async def list_conversations(
    session_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    List recent conversations.
    
    Optionally filter by session_id for user-specific conversations.
    """
    if not config.POSTGRES_URL:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured."
        )
    
    try:
        pg_db = get_postgres_db()
        conversations = await pg_db.get_recent_conversations(
            session_id=session_id,
            limit=limit
        )
        
        return {"conversations": conversations, "count": len(conversations)}
    except Exception as e:
        logger.error(f"List conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/generate", tags=["RAG"])
async def generate_embeddings(
    limit: int = Query(100, ge=1, le=1000, description="Maximum chunks to process")
):
    """
    Generate embeddings for pending chunks.
    
    Processes chunks that don't have embeddings yet.
    Run this after indexing to enable RAG functionality.
    """
    try:
        success, failed = await embedding_service.process_pending_chunks(limit=limit)
        
        stats = db.get_embedding_stats()
        
        return {
            "status": "completed",
            "processed": success,
            "failed": failed,
            "pending_remaining": stats["pending_chunks"],
        }
    except ValueError as e:
        logger.error(f"Embedding configuration error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM not configured: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


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
