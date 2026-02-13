"""
Second Brain Daemon - FastAPI Application
A local daemon that indexes an Obsidian vault and exposes a search API.

This module creates the FastAPI app, registers middleware and routers,
and manages the application lifespan (startup / shutdown).

All route handlers live in ``app.api.routes.*``.
All request/response models live in ``app.api.models.*``.
All business logic lives in ``app.services.*``.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import config
from app.db import db
from app.indexer import indexer
from app.watcher import watcher
from app.api.middleware import register_middleware
from app.api.routes import all_routers

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown."""
    logger.info("Starting Second Brain daemon...")

    try:
        config.validate()
        logger.info("Vault path: %s", config.VAULT_PATH)

        db.initialize()

        stats = db.get_stats()
        if stats["file_count"] == 0:
            logger.info("First run detected - performing full vault scan")
            indexed, errors = indexer.full_scan()
            logger.info("Initial scan complete: %d files indexed, %d errors", indexed, errors)
        else:
            logger.info("Existing index found - performing incremental scan")
            indexed, errors = indexer.incremental_scan()
            if indexed > 0:
                logger.info("Incremental scan: %d files updated, %d errors", indexed, errors)

        watcher.start()
        logger.info("Second Brain daemon started successfully")

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        raise
    except Exception as e:
        logger.error("Startup error: %s", e)
        raise

    yield

    logger.info("Shutting down Second Brain daemon...")
    watcher.stop()
    db.close()
    logger.info("Second Brain daemon stopped")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Second Brain API",
    description="Local API for searching and accessing an indexed Obsidian vault",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware (CORS + API-key auth)
register_middleware(app)

# Register all domain routers
for router in all_routers:
    app.include_router(router)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid request", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )
