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
