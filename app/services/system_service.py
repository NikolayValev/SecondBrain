"""
System service: health checks, stats, and provider configuration.
"""

import logging

from app.config import config, Config
from app.db import db
from app.embeddings import embedding_service
from app.watcher import watcher
from app import llm
from app.rag_techniques import list_techniques
from app.api.models.system import (
    ProviderModel,
    ProviderInfo,
    ProviderStatus,
    VectorStoreStatus,
    HealthResponse,
    StatsResponse,
    RAGTechniqueInfo,
    ConfigDefaults,
    ConfigResponse,
)

logger = logging.getLogger(__name__)


class SystemService:
    """Handles system-level operations: health, stats, config."""

    async def get_health(self) -> HealthResponse:
        """Build the health check response with provider statuses."""
        providers = {}
        for provider_name in ["ollama", "openai", "gemini", "anthropic"]:
            status = await llm.check_provider_availability(provider_name)
            if status["available"]:
                models_loaded = None
                if provider_name == "ollama":
                    models_loaded = await llm.list_ollama_models()
                providers[provider_name] = ProviderStatus(
                    available=True,
                    models_loaded=models_loaded,
                )
            else:
                providers[provider_name] = ProviderStatus(
                    available=False,
                    error=status.get("error"),
                )

        embedding_stats = db.get_embedding_stats()
        vector_store = VectorStoreStatus(
            type="sqlite",
            documents_indexed=embedding_stats.get("embedding_count", 0),
        )

        return HealthResponse(
            status="ok",
            version="1.0.0",
            vault_path=str(config.VAULT_PATH),
            watcher_running=watcher.is_running,
            providers=providers,
            vector_store=vector_store,
        )

    def get_stats(self) -> StatsResponse:
        """Return database statistics."""
        stats = db.get_stats()
        return StatsResponse(
            file_count=stats["file_count"],
            section_count=stats["section_count"],
            tag_count=stats["tag_count"],
            link_count=stats["link_count"],
            last_indexed=stats["last_indexed"],
        )

    async def get_config(self) -> ConfigResponse:
        """Build the full configuration response."""
        providers = []

        # --- Ollama ---
        ollama_status = await llm.check_provider_availability("ollama")
        ollama_models = []
        if ollama_status["available"]:
            model_names = await llm.list_ollama_models()
            for name in model_names:
                ollama_models.append(ProviderModel(
                    id=name,
                    name=name.title().replace("-", " "),
                    context_length=128000,
                    available=True,
                ))
        providers.append(ProviderInfo(
            id="ollama",
            name="Ollama (Local)",
            available=ollama_status["available"],
            base_url=Config.OLLAMA_BASE_URL,
            models=ollama_models,
            error=ollama_status.get("error") if not ollama_status["available"] else None,
        ))

        # --- OpenAI ---
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
            error=openai_status.get("error") if not openai_status["available"] else None,
        ))

        # --- Gemini ---
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
            error=gemini_status.get("error") if not gemini_status["available"] else None,
        ))

        # --- Anthropic ---
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
            error=anthropic_status.get("error") if not anthropic_status["available"] else None,
        ))

        # RAG techniques
        rag_techniques = [RAGTechniqueInfo(**t) for t in list_techniques()]

        # Defaults
        default_provider = Config.LLM_PROVIDER.lower()
        default_model = {
            "openai": Config.OPENAI_MODEL,
            "gemini": Config.GEMINI_MODEL,
            "ollama": Config.OLLAMA_MODEL,
            "anthropic": Config.ANTHROPIC_MODEL,
        }.get(default_provider, "gpt-4o")

        embedding_model = embedding_service.get_model_name()

        return ConfigResponse(
            providers=providers,
            rag_techniques=rag_techniques,
            defaults=ConfigDefaults(
                provider=default_provider,
                model=default_model,
                rag_technique="basic",
            ),
            embedding_model=embedding_model,
            vector_store="sqlite",
        )


# Singleton
system_service = SystemService()
