# Second Brain - AI Coding Instructions

## Architecture Overview

Second Brain is a **local daemon** that indexes an Obsidian vault and exposes a FastAPI REST API. Key architecture:

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│  Obsidian Vault │────▶│  SQLite (FTS5)│────▶│  PostgreSQL     │
│  (markdown)     │     │  (local)      │     │  (Prisma/Next.js)│
└─────────────────┘     └───────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌───────────────┐
│  File Watcher   │     │  RAG Pipeline │
│  (watchdog)     │     │  (LLM + embed)│
└─────────────────┘     └───────────────┘
```

**Data Flow**: Files → `parser.py` → `indexer.py` → `db.py` (SQLite) → `sync_service.py` → `db_postgres.py` (PostgreSQL)

**RAG Pipeline**: Query → `embeddings.py` → `vector_search.py` → `llm.py` → `rag.py`

## Key Modules & Patterns

| Module | Responsibility | Singleton Instance |
|--------|---------------|-------------------|
| [app/db.py](app/db.py) | SQLite with FTS5 full-text search | `db` |
| [app/db_postgres.py](app/db_postgres.py) | Async PostgreSQL (asyncpg) for Prisma | `get_postgres_db()` |
| [app/indexer.py](app/indexer.py) | File indexing orchestration | `indexer` |
| [app/watcher.py](app/watcher.py) | Debounced file system monitoring | `watcher` |
| [app/llm.py](app/llm.py) | Multi-provider LLM abstraction | `get_llm_provider()` |
| [app/embeddings.py](app/embeddings.py) | Embedding generation & storage | `embedding_service` |
| [app/rag.py](app/rag.py) | RAG Q&A orchestration | `rag_service` |

**Pattern**: Most modules export a singleton instance (lowercase) alongside the class. Use the singleton for production, instantiate the class directly for testing with dependency injection.

## LLM Provider Pattern

The codebase uses a **strategy pattern** for LLM providers. All providers extend `LLMProvider` ABC:

```python
# In app/llm.py - use get_llm_provider() to get configured provider
provider = llm.get_llm_provider()  # Returns OpenAI/Gemini/Ollama based on LLM_PROVIDER
await provider.chat(messages=[{"role": "user", "content": "..."}])
await provider.embed("text")  # Returns list[float]
```

Configured via `.env`: `LLM_PROVIDER=gemini|openai|ollama`

## Dual Database Architecture

- **SQLite** (`db.py`): Primary local storage with FTS5 for full-text search. Sync operations.
- **PostgreSQL** (`db_postgres.py`): Remote sync target for Next.js/Prisma consumption. Fully async with asyncpg.

The Prisma schema in [prisma/schema.prisma](prisma/schema.prisma) defines the PostgreSQL structure. Sync is handled by `sync_service.py`.

## Commands & Workflows

```powershell
# Development server (with hot reload)
python -m uvicorn app.main:app --reload

# Run tests
pytest                          # All tests
pytest tests/test_api.py -v     # Specific file
pytest -k "test_search"         # Pattern match

# Direct module run
python -m app.main
```

## Test Fixtures

Tests use fixtures from [tests/conftest.py](tests/conftest.py):
- `temp_dir` / `temp_vault`: Isolated temp directories with sample markdown
- `test_db`: Fresh Database instance with temp path
- `test_indexer`: Indexer wired to test fixtures

**Async tests**: Use `pytest.mark.asyncio` (mode set to `auto` in pytest.ini).

## Embedding Storage Format

Embeddings are stored as packed binary floats:
```python
# Encode: list[float] → bytes
embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)
# Decode: bytes → list[float]
embedding = struct.unpack(f'{num_floats}f', data)
```

## Inbox Processor Classification

The inbox processor ([app/inbox_processor.py](app/inbox_processor.py)) uses a pluggable classification system:
- `ClassificationMethod.RULES_ONLY`: Regex patterns on title/content/tags
- `ClassificationMethod.LLM_ONLY`: LLM-based classification
- `ClassificationMethod.RULES_THEN_LLM`: Rules first, LLM fallback

Configure via `inbox_config.yaml`.

## Environment Variables

Key variables in `.env`:
```ini
VAULT_PATH=C:\path\to\obsidian\vault  # Required
LLM_PROVIDER=gemini|openai|ollama
GEMINI_API_KEY=...                    # Required if using Gemini
DATABASE_URL=postgres://...           # For Prisma sync
```

## API Endpoints Pattern

All endpoints are defined in [app/main.py](app/main.py) with Pydantic models for request/response. Follow existing patterns:
- Use `BaseModel` for all request/response types
- Return `JSONResponse` for errors with appropriate status codes
- Async endpoints for any I/O operations
