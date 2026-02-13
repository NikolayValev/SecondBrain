"""
System-related API models: health, stats, config, providers.
"""

from typing import Optional

from pydantic import BaseModel


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


class StatsResponse(BaseModel):
    """Database statistics response."""
    file_count: int
    section_count: int
    tag_count: int
    link_count: int
    last_indexed: Optional[str]


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


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
