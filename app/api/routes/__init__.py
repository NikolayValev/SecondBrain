"""
Route modules for the Second Brain API.

Each module defines a FastAPI APIRouter for a specific domain.
All routers are collected in ``all_routers`` for easy inclusion.
"""

from app.api.routes.system import router as system_router
from app.api.routes.search import router as search_router
from app.api.routes.files import router as files_router
from app.api.routes.rag import router as rag_router
from app.api.routes.inbox import router as inbox_router
from app.api.routes.sync import router as sync_router
from app.api.routes.indexing import router as indexing_router
from app.api.routes.conversations import router as conversations_router

all_routers = [
    system_router,
    search_router,
    files_router,
    rag_router,
    inbox_router,
    sync_router,
    indexing_router,
    conversations_router,
]

__all__ = ["all_routers"]
