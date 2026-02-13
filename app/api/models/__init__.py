"""
Pydantic models (schemas) for API request/response types.

All models are re-exported here for convenience:
    from app.api.models import SearchResponse, AskRequest, ...
"""

from app.api.models.system import (
    HealthResponse,
    StatsResponse,
    ProviderModel,
    ProviderInfo,
    ProviderStatus,
    VectorStoreStatus,
    RAGTechniqueInfo,
    ConfigDefaults,
    ConfigResponse,
    ErrorResponse,
)
from app.api.models.search import (
    SearchResult,
    SearchResponse,
    SemanticSearchRequest,
    SemanticSearchResult,
    SemanticSearchResponse,
)
from app.api.models.files import (
    FileResponse,
    BacklinkItem,
    BacklinksResponse,
    TagItem,
    TagsResponse,
)
from app.api.models.rag import (
    AskRequest,
    Source,
    TokenUsage,
    AskResponse,
    SourceInfo,
    EmbeddingStatsResponse,
)
from app.api.models.inbox import (
    InboxProcessRequest,
    InboxFileResult,
    InboxProcessResponse,
    InboxFileInfo,
    InboxFolderInfo,
    InboxContentsResponse,
)
from app.api.models.sync import (
    SyncRequest,
    SyncResponse,
    PostgresStatsResponse,
)
from app.api.models.conversations import (
    ConversationCreate,
    MessageCreate,
    ConversationResponse,
)
from app.api.models.indexing import (
    IndexRequest,
    IndexResponse,
    IndexStatusResponse,
)

__all__ = [
    # System
    "HealthResponse",
    "StatsResponse",
    "ProviderModel",
    "ProviderInfo",
    "ProviderStatus",
    "VectorStoreStatus",
    "RAGTechniqueInfo",
    "ConfigDefaults",
    "ConfigResponse",
    "ErrorResponse",
    # Search
    "SearchResult",
    "SearchResponse",
    "SemanticSearchRequest",
    "SemanticSearchResult",
    "SemanticSearchResponse",
    # Files
    "FileResponse",
    "BacklinkItem",
    "BacklinksResponse",
    "TagItem",
    "TagsResponse",
    # RAG
    "AskRequest",
    "Source",
    "TokenUsage",
    "AskResponse",
    "SourceInfo",
    "EmbeddingStatsResponse",
    # Inbox
    "InboxProcessRequest",
    "InboxFileResult",
    "InboxProcessResponse",
    "InboxFileInfo",
    "InboxFolderInfo",
    "InboxContentsResponse",
    # Sync
    "SyncRequest",
    "SyncResponse",
    "PostgresStatsResponse",
    # Conversations
    "ConversationCreate",
    "MessageCreate",
    "ConversationResponse",
    # Indexing
    "IndexRequest",
    "IndexResponse",
    "IndexStatusResponse",
]
