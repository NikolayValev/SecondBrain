"""
Second Brain Daemon - FastAPI Application
A local daemon that indexes an Obsidian vault and exposes a search API.
"""

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.config import config, Config
from app.db import db
from app.indexer import indexer
from app.watcher import watcher
from app.rag import rag_service
from app.embeddings import embedding_service
from app.inbox_processor import inbox_processor, InboxConfig, create_default_config
from app.sync_service import get_sync_service, run_sync
from app.db_postgres import get_postgres_db, close_postgres_db
from app import llm
from app.rag_techniques import get_technique, list_techniques
from app.vector_search import vector_search

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from X-API-Key header.
    If API_KEY is not configured, authentication is disabled (development mode).
    """
    # If no API key configured, allow all requests (dev mode)
    if not config.API_KEY:
        return "dev-mode"
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return api_key


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
    code: Optional[str] = None


class AskRequest(BaseModel):
    """Request body for /ask endpoint (new API spec)."""
    question: str
    conversation_id: Optional[str] = None
    provider: str = "gemini"
    model: Optional[str] = None
    rag_technique: str = "basic"
    include_sources: bool = True


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


class AskResponse(BaseModel):
    """Response from /ask endpoint (new API spec)."""
    answer: str
    sources: list[Source]
    conversation_id: Optional[str] = None
    model_used: str
    tokens_used: Optional[TokenUsage] = None


# Legacy response model for backward compatibility
class LegacyAskRequest(BaseModel):
    """Legacy request body for /ask endpoint."""
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


# New models for API spec
class ProviderModel(BaseModel):
    """Model information within a provider."""
    id: str
    name: str
    context_length: int = 128000
    available: bool = True


class ProviderInfo(BaseModel):
    """Provider information."""
    id: str
    name: str
    available: bool
    base_url: Optional[str] = None
    models: list[ProviderModel] = []
    error: Optional[str] = None


class RAGTechniqueInfo(BaseModel):
    """RAG technique information."""
    id: str
    name: str
    description: str


class ConfigDefaults(BaseModel):
    """Default configuration values."""
    provider: str
    model: str
    rag_technique: str


class ConfigResponse(BaseModel):
    """Response for /config endpoint."""
    providers: list[ProviderInfo]
    rag_techniques: list[RAGTechniqueInfo]
    defaults: ConfigDefaults
    embedding_model: str
    vector_store: str = "sqlite"


class ProviderStatus(BaseModel):
    """Provider status for health check."""
    available: bool
    models_loaded: Optional[list[str]] = None
    error: Optional[str] = None


class VectorStoreStatus(BaseModel):
    """Vector store status for health check."""
    type: str
    documents_indexed: int


class HealthResponse(BaseModel):
    """Enhanced health check response."""
    status: str
    version: str = "1.0.0"
    vault_path: str
    watcher_running: bool
    providers: dict[str, ProviderStatus] = {}
    vector_store: Optional[VectorStoreStatus] = None


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


# State for indexing jobs
_indexing_jobs: dict[str, dict] = {}


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

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"/health", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def api_key_middleware(request, call_next):
    """
    Middleware to verify API key for all endpoints except public ones.
    If API_KEY is not configured, all requests are allowed (dev mode).
    """
    # Skip authentication for public endpoints
    if request.url.path in PUBLIC_ENDPOINTS:
        return await call_next(request)
    
    # If no API key configured, allow all requests (dev mode)
    if not config.API_KEY:
        return await call_next(request)
    
    # Check API key
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing API key. Include 'X-API-Key' header."},
        )
    
    if api_key != config.API_KEY:
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid API key"},
        )
    
    return await call_next(request)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Enhanced health check endpoint.
    Returns the status of the daemon, providers, and vector store.
    """
    # Check provider availability
    providers = {}
    for provider_name in ["ollama", "openai", "gemini", "anthropic"]:
        status = await llm.check_provider_availability(provider_name)
        if status["available"]:
            models_loaded = None
            if provider_name == "ollama":
                models_loaded = await llm.list_ollama_models()
            providers[provider_name] = ProviderStatus(
                available=True,
                models_loaded=models_loaded
            )
        else:
            providers[provider_name] = ProviderStatus(
                available=False,
                error=status.get("error")
            )
    
    # Get vector store stats
    stats = db.get_stats()
    embedding_stats = db.get_embedding_stats()
    vector_store = VectorStoreStatus(
        type="sqlite",
        documents_indexed=embedding_stats.get("embedding_count", 0)
    )
    
    return HealthResponse(
        status="ok",
        version="1.0.0",
        vault_path=str(config.VAULT_PATH),
        watcher_running=watcher.is_running,
        providers=providers,
        vector_store=vector_store
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


@app.get("/config", response_model=ConfigResponse, tags=["System"])
async def get_config():
    """
    Get available configuration options.
    Returns available providers, models, and RAG techniques.
    """
    providers = []
    
    # Ollama provider
    ollama_status = await llm.check_provider_availability("ollama")
    ollama_models = []
    if ollama_status["available"]:
        model_names = await llm.list_ollama_models()
        for name in model_names:
            ollama_models.append(ProviderModel(
                id=name,
                name=name.title().replace("-", " "),
                context_length=128000,
                available=True
            ))
    providers.append(ProviderInfo(
        id="ollama",
        name="Ollama (Local)",
        available=ollama_status["available"],
        base_url=Config.OLLAMA_BASE_URL,
        models=ollama_models,
        error=ollama_status.get("error") if not ollama_status["available"] else None
    ))
    
    # OpenAI provider
    openai_status = await llm.check_provider_availability("openai")
    openai_models = [
        ProviderModel(id="gpt-4o", name="GPT-4o", context_length=128000),
        ProviderModel(id="gpt-4o-mini", name="GPT-4o Mini", context_length=128000),
        ProviderModel(id="gpt-4-turbo", name="GPT-4 Turbo", context_length=128000),
        ProviderModel(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", context_length=16385),
    ] if openai_status["available"] else []
    providers.append(ProviderInfo(
        id="openai",
        name="OpenAI",
        available=openai_status["available"],
        models=openai_models,
        error=openai_status.get("error") if not openai_status["available"] else None
    ))
    
    # Gemini provider
    gemini_status = await llm.check_provider_availability("gemini")
    gemini_models = [
        ProviderModel(id="gemini-2.5-flash", name="Gemini 2.5 Flash", context_length=1000000),
        ProviderModel(id="gemini-2.5-pro", name="Gemini 2.5 Pro", context_length=1000000),
        ProviderModel(id="gemini-2.0-flash", name="Gemini 2.0 Flash", context_length=1000000),
        ProviderModel(id="gemini-1.5-pro", name="Gemini 1.5 Pro", context_length=2000000),
        ProviderModel(id="gemini-1.5-flash", name="Gemini 1.5 Flash", context_length=1000000),
    ] if gemini_status["available"] else []
    providers.append(ProviderInfo(
        id="gemini",
        name="Google Gemini",
        available=gemini_status["available"],
        models=gemini_models,
        error=gemini_status.get("error") if not gemini_status["available"] else None
    ))
    
    # Anthropic provider
    anthropic_status = await llm.check_provider_availability("anthropic")
    anthropic_models = [
        ProviderModel(id="claude-sonnet-4-20250514", name="Claude Sonnet 4", context_length=200000),
        ProviderModel(id="claude-3-5-sonnet-20241022", name="Claude 3.5 Sonnet", context_length=200000),
        ProviderModel(id="claude-3-opus-20240229", name="Claude 3 Opus", context_length=200000),
        ProviderModel(id="claude-3-haiku-20240307", name="Claude 3 Haiku", context_length=200000),
    ] if anthropic_status["available"] else []
    providers.append(ProviderInfo(
        id="anthropic",
        name="Anthropic",
        available=anthropic_status["available"],
        models=anthropic_models,
        error=anthropic_status.get("error") if not anthropic_status["available"] else None
    ))
    
    # Get RAG techniques
    rag_techniques = [
        RAGTechniqueInfo(**t) for t in list_techniques()
    ]
    
    # Determine defaults
    default_provider = Config.LLM_PROVIDER.lower()
    default_model = {
        "openai": Config.OPENAI_MODEL,
        "gemini": Config.GEMINI_MODEL,
        "ollama": Config.OLLAMA_MODEL,
        "anthropic": Config.ANTHROPIC_MODEL,
    }.get(default_provider, "gpt-4o")
    
    # Get embedding model
    embedding_model = embedding_service.get_model_name()
    
    return ConfigResponse(
        providers=providers,
        rag_techniques=rag_techniques,
        defaults=ConfigDefaults(
            provider=default_provider,
            model=default_model,
            rag_technique="basic"
        ),
        embedding_model=embedding_model,
        vector_store="sqlite"
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


@app.post("/search", response_model=SemanticSearchResponse, tags=["Search"])
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search across the vault (without LLM generation).
    
    Uses embeddings for semantic similarity matching.
    Supports different RAG techniques for retrieval.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    import time
    
    try:
        # Get the RAG technique
        technique = get_technique(request.rag_technique)
        
        # Time the embedding
        embed_start = time.time()
        
        # Retrieve using the technique
        rag_context = await technique.retrieve(
            query=request.query,
            top_k=request.limit,
            threshold=0.3,
        )
        
        embed_time = (time.time() - embed_start) * 1000
        
        # Build results
        results = []
        for chunk in rag_context.chunks:
            # Get file metadata
            file_record = db.get_file_by_path(chunk.file_path)
            metadata = {}
            if file_record:
                # Get tags for file
                with db.cursor() as cur:
                    cur.execute("""
                        SELECT t.name FROM tags t
                        JOIN file_tags ft ON t.id = ft.tag_id
                        WHERE ft.file_id = ?
                    """, (chunk.file_id,))
                    tags = [row["name"] for row in cur.fetchall()]
                    metadata["tags"] = tags
                    metadata["created_at"] = file_record.get("created_at")
                    metadata["updated_at"] = file_record.get("updated_at")
            
            results.append(SemanticSearchResult(
                path=chunk.file_path,
                title=chunk.file_title,
                snippet=chunk.chunk_content[:300] + "..." if len(chunk.chunk_content) > 300 else chunk.chunk_content,
                score=round(chunk.similarity, 4),
                metadata=metadata,
            ))
        
        return SemanticSearchResponse(
            results=results,
            query_embedding_time_ms=round(embed_time, 2),
            search_time_ms=round(embed_time, 2),  # Total time for now
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
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


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
async def trigger_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Trigger document indexing for the knowledge base.
    
    Runs indexing in the background and returns a job ID for tracking.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    _indexing_jobs[job_id] = {
        "status": "started",
        "progress": 0,
        "documents_processed": 0,
        "documents_total": 0,
        "started_at": datetime.now().isoformat(),
    }
    
    async def run_indexing():
        try:
            _indexing_jobs[job_id]["status"] = "indexing"
            
            if request.paths:
                # Index specific paths
                total = len(request.paths)
                processed = 0
                errors = 0
                
                for path in request.paths:
                    try:
                        # Reindex specific file
                        full_path = config.VAULT_PATH / path
                        if full_path.exists():
                            indexer.index_file(str(full_path), force=request.force)
                            processed += 1
                        else:
                            errors += 1
                    except Exception as e:
                        logger.error(f"Failed to index {path}: {e}")
                        errors += 1
                    
                    _indexing_jobs[job_id]["documents_processed"] = processed
                    _indexing_jobs[job_id]["progress"] = processed / total if total > 0 else 1
                
                _indexing_jobs[job_id]["documents_total"] = total
            else:
                # Full or incremental scan
                if request.force:
                    indexed, errors = indexer.full_scan()
                else:
                    indexed, errors = indexer.incremental_scan()
                
                _indexing_jobs[job_id]["documents_processed"] = indexed
                _indexing_jobs[job_id]["documents_total"] = indexed + errors
            
            _indexing_jobs[job_id]["status"] = "completed"
            _indexing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Indexing job {job_id} failed: {e}")
            _indexing_jobs[job_id]["status"] = "error"
            _indexing_jobs[job_id]["error"] = str(e)
    
    # Queue the background task
    background_tasks.add_task(run_indexing)
    
    # Count documents to queue
    if request.paths:
        docs_queued = len(request.paths)
    else:
        # Count markdown files in vault
        docs_queued = sum(1 for _ in config.VAULT_PATH.rglob("*.md"))
    
    _indexing_jobs[job_id]["documents_total"] = docs_queued
    
    return IndexResponse(
        status="started",
        job_id=job_id,
        documents_queued=docs_queued,
    )


@app.get("/index/status", response_model=IndexStatusResponse, tags=["Indexing"])
async def get_index_status():
    """
    Get the current indexing status.
    """
    stats = db.get_stats()
    embedding_stats = db.get_embedding_stats()
    
    # Find any active job
    current_job = None
    for job_id, job in _indexing_jobs.items():
        if job["status"] in ("started", "indexing"):
            current_job = {
                "job_id": job_id,
                "progress": job.get("progress", 0),
                "documents_processed": job.get("documents_processed", 0),
                "documents_total": job.get("documents_total", 0),
            }
            break
    
    # Determine overall status
    if current_job:
        status = "indexing"
    else:
        status = "idle"
    
    return IndexStatusResponse(
        status=status,
        documents_indexed=stats["file_count"],
        documents_pending=embedding_stats.get("pending_chunks", 0),
        last_indexed_at=stats.get("last_indexed"),
        current_job=current_job,
    )


# RAG Endpoints

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(request: AskRequest):
    """
    Ask a question using RAG (Retrieval-Augmented Generation).
    
    Searches the knowledge base for relevant content and uses an LLM
    to generate an answer based on the retrieved context.
    
    Supports multiple providers, models, and RAG techniques.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get the RAG technique
        technique = get_technique(request.rag_technique)
        
        # Retrieve context using the selected technique
        rag_context = await technique.retrieve(
            query=request.question,
            top_k=5,
            threshold=0.3,
        )
        
        if not rag_context.chunks:
            return AskResponse(
                answer="I couldn't find any relevant information in your knowledge base to answer this question.",
                sources=[],
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                model_used=request.model or Config.GEMINI_MODEL,
            )
        
        # Get the LLM provider
        try:
            provider = llm.get_provider_by_name(request.provider)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
        
        # Build the prompt
        system_prompt = """You are a helpful assistant that answers questions based on the user's personal knowledge base (Obsidian vault).

Use the provided context from the knowledge base to answer the question. If the context doesn't contain relevant information, say so honestly.

Guidelines:
- Be concise but thorough
- Reference specific notes when relevant
- If information is incomplete, acknowledge it
- Don't make up information not in the context"""
        
        user_message = f"""Context from knowledge base:
---
{rag_context.context_text}
---

Question: {request.question}

Answer based on the context above:"""
        
        # Generate answer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Determine the model to use
        model_used = request.model
        if not model_used:
            model_used = {
                "openai": Config.OPENAI_MODEL,
                "gemini": Config.GEMINI_MODEL,
                "ollama": Config.OLLAMA_MODEL,
                "anthropic": Config.ANTHROPIC_MODEL,
            }.get(request.provider.lower(), Config.GEMINI_MODEL)
        
        answer = await provider.chat(messages=messages)
        
        # Build sources list
        sources = []
        if request.include_sources:
            seen_files = set()
            for chunk in rag_context.chunks:
                if chunk.file_path not in seen_files:
                    sources.append(Source(
                        path=chunk.file_path,
                        title=chunk.file_title,
                        snippet=chunk.chunk_content[:200] + "..." if len(chunk.chunk_content) > 200 else chunk.chunk_content,
                        score=round(chunk.similarity, 3),
                    ))
                    seen_files.add(chunk.file_path)
        
        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        return AskResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            model_used=model_used,
            tokens_used=TokenUsage(prompt=0, completion=0, total=0),  # TODO: Track actual usage
        )
        
    except ValueError as e:
        logger.error(f"RAG configuration error: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "code": "RAG_ERROR"}
        )
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate answer: {str(e)}", "code": "GENERATION_ERROR"}
        )


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


@app.get("/inbox/contents", response_model=InboxContentsResponse, tags=["Inbox"])
async def get_inbox_contents():
    """
    Get the full contents of the inbox folder including all subfolders.
    
    Returns a hierarchical view of all files and folders in 00_Inbox.
    """
    inbox_path = config.VAULT_PATH / "00_Inbox"
    
    if not inbox_path.exists():
        raise HTTPException(status_code=404, detail="Inbox folder (00_Inbox) not found")
    
    def get_file_info(file_path: Path) -> InboxFileInfo:
        """Create file info from path."""
        stat = file_path.stat()
        return InboxFileInfo(
            name=file_path.name,
            path=str(file_path.relative_to(config.VAULT_PATH)),
            size_bytes=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        )
    
    def scan_folder(folder_path: Path) -> tuple[list[InboxFileInfo], list[InboxFolderInfo]]:
        """Recursively scan a folder for files and subfolders."""
        files = []
        folders = []
        
        try:
            for item in sorted(folder_path.iterdir()):
                if item.name.startswith('.'):
                    continue  # Skip hidden files/folders
                
                if item.is_file():
                    if item.suffix.lower() in ('.md', '.markdown'):
                        files.append(get_file_info(item))
                elif item.is_dir():
                    sub_files, sub_folders = scan_folder(item)
                    folder_info = InboxFolderInfo(
                        name=item.name,
                        path=str(item.relative_to(config.VAULT_PATH)),
                        files=sub_files,
                        folders=sub_folders,
                    )
                    folders.append(folder_info)
        except PermissionError:
            logger.warning(f"Permission denied accessing: {folder_path}")
        
        return files, folders
    
    root_files, sub_folders = scan_folder(inbox_path)
    
    # Count total files and folders
    def count_items(folders: list[InboxFolderInfo]) -> tuple[int, int]:
        total_files = 0
        total_folders = len(folders)
        for folder in folders:
            total_files += len(folder.files)
            sub_files, sub_folder_count = count_items(folder.folders)
            total_files += sub_files
            total_folders += sub_folder_count
        return total_files, total_folders
    
    nested_files, nested_folders = count_items(sub_folders)
    
    return InboxContentsResponse(
        inbox_path=str(inbox_path.relative_to(config.VAULT_PATH)),
        total_files=len(root_files) + nested_files,
        total_folders=nested_folders,
        root_files=root_files,
        folders=sub_folders,
    )


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
