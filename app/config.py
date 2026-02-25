"""
Configuration module for Second Brain daemon.
Loads settings from environment variables or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (parent of app/ directory)
_project_root = Path(__file__).parent.parent
_env_path = _project_root / ".env"
load_dotenv(_env_path)


def _get_bool(name: str, default: str = "false") -> bool:
    """Parse environment variable into bool."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: str) -> int:
    """Parse environment variable into int with fallback."""
    raw = os.getenv(name, default).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _get_csv(name: str, default: list[str]) -> list[str]:
    """Parse comma-separated env var into a non-empty list."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


class Config:
    """Application configuration."""
    
    # Vault path - must be set via environment variable or .env
    VAULT_PATH: Path = Path(os.getenv("VAULT_PATH", ""))
    
    # Database settings
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", "./second_brain.db"))
    
    # PostgreSQL settings (for Next.js/Prisma integration)
    POSTGRES_URL: str = os.getenv(
        "DATABASE_URL",  # Matches Prisma convention
        os.getenv("POSTGRES_URL", "")
    )
    POSTGRES_SYNC_ENABLED: bool = os.getenv("POSTGRES_SYNC_ENABLED", "false").lower() == "true"
    POSTGRES_SYNC_ON_CHANGE: bool = os.getenv("POSTGRES_SYNC_ON_CHANGE", "false").lower() == "true"
    
    # File watcher settings
    DEBOUNCE_SECONDS: float = float(os.getenv("DEBOUNCE_SECONDS", "1.0"))
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    BRAIN_API_KEY: str = os.getenv("BRAIN_API_KEY", "")  # Required for authenticated endpoints
    REQUIRE_API_KEY: bool = _get_bool("REQUIRE_API_KEY", "false")
    EXPOSE_API_DOCS: bool = _get_bool("EXPOSE_API_DOCS", "false")
    EXPOSE_CONFIG_PUBLIC: bool = _get_bool("EXPOSE_CONFIG_PUBLIC", "false")
    DEBUG: bool = _get_bool("DEBUG", "false")
    PUBLIC_API_MODE: bool = _get_bool("PUBLIC_API_MODE", "false")
    CORS_ALLOW_CREDENTIALS: bool = _get_bool("CORS_ALLOW_CREDENTIALS", "false")
    ALLOWED_ORIGINS: list[str] = _get_csv(
        "ALLOWED_ORIGINS",
        ["http://localhost:3000", "http://127.0.0.1:3000"],
    )
    ALLOWED_HOSTS: list[str] = _get_csv(
        "ALLOWED_HOSTS",
        ["127.0.0.1", "localhost"],
    )
    RATE_LIMIT_ENABLED: bool = _get_bool("RATE_LIMIT_ENABLED", "true")
    RATE_LIMIT_WINDOW_SECONDS: int = _get_int("RATE_LIMIT_WINDOW_SECONDS", "60")
    RATE_LIMIT_DEFAULT_PER_WINDOW: int = _get_int("RATE_LIMIT_DEFAULT_PER_WINDOW", "120")
    RATE_LIMIT_ASK_PER_WINDOW: int = _get_int("RATE_LIMIT_ASK_PER_WINDOW", "30")
    RATE_LIMIT_EMBEDDINGS_PER_WINDOW: int = _get_int("RATE_LIMIT_EMBEDDINGS_PER_WINDOW", "6")
    RATE_LIMIT_SYNC_PER_WINDOW: int = _get_int("RATE_LIMIT_SYNC_PER_WINDOW", "20")
    RATE_LIMIT_INDEXING_PER_WINDOW: int = _get_int("RATE_LIMIT_INDEXING_PER_WINDOW", "20")
    MAX_REQUEST_BYTES: int = _get_int("MAX_REQUEST_BYTES", "1048576")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # File extensions to index
    MARKDOWN_EXTENSIONS: tuple = (".md", ".markdown")
    
    # LLM Provider settings
    # Supported providers: "openai", "gemini", "ollama"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Google Gemini settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    
    # Ollama settings (local models)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Anthropic settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    
    # Embedding provider override (for providers like Anthropic that lack embedding APIs)
    # If set, embeddings use this provider instead of the chat LLM_PROVIDER.
    # Valid values: "openai", "gemini", "ollama", or empty (use LLM_PROVIDER).
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "")
    
    # Common LLM settings
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Reranker settings (for rerank RAG technique)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.VAULT_PATH or not cls.VAULT_PATH.exists():
            raise ValueError(
                f"VAULT_PATH must be set to a valid directory. "
                f"Current value: {cls.VAULT_PATH}"
            )
        if not cls.VAULT_PATH.is_dir():
            raise ValueError(f"VAULT_PATH must be a directory: {cls.VAULT_PATH}")
        if cls.REQUIRE_API_KEY and not cls.BRAIN_API_KEY:
            raise ValueError("REQUIRE_API_KEY=true but BRAIN_API_KEY is not set")
        if cls.REQUIRE_API_KEY and len(cls.BRAIN_API_KEY) < 24:
            raise ValueError("BRAIN_API_KEY must be at least 24 characters when REQUIRE_API_KEY=true")
        if cls.RATE_LIMIT_WINDOW_SECONDS <= 0:
            raise ValueError("RATE_LIMIT_WINDOW_SECONDS must be > 0")
        for name, value in (
            ("RATE_LIMIT_DEFAULT_PER_WINDOW", cls.RATE_LIMIT_DEFAULT_PER_WINDOW),
            ("RATE_LIMIT_ASK_PER_WINDOW", cls.RATE_LIMIT_ASK_PER_WINDOW),
            ("RATE_LIMIT_EMBEDDINGS_PER_WINDOW", cls.RATE_LIMIT_EMBEDDINGS_PER_WINDOW),
            ("RATE_LIMIT_SYNC_PER_WINDOW", cls.RATE_LIMIT_SYNC_PER_WINDOW),
            ("RATE_LIMIT_INDEXING_PER_WINDOW", cls.RATE_LIMIT_INDEXING_PER_WINDOW),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if cls.MAX_REQUEST_BYTES <= 0:
            raise ValueError("MAX_REQUEST_BYTES must be > 0")
    
    @classmethod
    def validate_llm_config(cls) -> None:
        """Validate LLM provider configuration."""
        provider = cls.LLM_PROVIDER.lower()
        
        if provider not in ("openai", "gemini", "ollama", "anthropic"):
            raise ValueError(
                f"LLM_PROVIDER must be one of: openai, gemini, ollama, anthropic. "
                f"Got: {provider}"
            )
        
        if provider == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI provider")
        
        if provider == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set when using Gemini provider")
        
        if provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set when using Anthropic provider")
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get configuration for the active LLM provider."""
        provider = cls.LLM_PROVIDER.lower()
        
        base_config = {
            "provider": provider,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
        }
        
        if provider == "openai":
            return {
                **base_config,
                "api_key": cls.OPENAI_API_KEY,
                "model": cls.OPENAI_MODEL,
                "base_url": cls.OPENAI_BASE_URL,
                "embedding_model": cls.OPENAI_EMBEDDING_MODEL,
            }
        elif provider == "gemini":
            return {
                **base_config,
                "api_key": cls.GEMINI_API_KEY,
                "model": cls.GEMINI_MODEL,
                "embedding_model": cls.GEMINI_EMBEDDING_MODEL,
            }
        elif provider == "ollama":
            return {
                **base_config,
                "base_url": cls.OLLAMA_BASE_URL,
                "model": cls.OLLAMA_MODEL,
                "embedding_model": cls.OLLAMA_EMBEDDING_MODEL,
            }
        elif provider == "anthropic":
            return {
                **base_config,
                "api_key": cls.ANTHROPIC_API_KEY,
                "model": cls.ANTHROPIC_MODEL,
            }
        
        raise ValueError(f"Unknown LLM provider: {provider}")


# Singleton config instance
config = Config()
