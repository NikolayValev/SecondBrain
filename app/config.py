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
    
    # File watcher settings
    DEBOUNCE_SECONDS: float = float(os.getenv("DEBOUNCE_SECONDS", "1.0"))
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # File extensions to index
    MARKDOWN_EXTENSIONS: tuple = (".md", ".markdown")
    
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


# Singleton config instance
config = Config()
