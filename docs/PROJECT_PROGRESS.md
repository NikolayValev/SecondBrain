# Second Brain Project Progress

## Overview

Second Brain is a Python-powered knowledge management system that indexes an Obsidian vault and provides intelligent search, RAG-powered Q&A with multiple LLM providers, and automated document processing. The system supports PostgreSQL sync for Next.js frontend integration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Second Brain System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Obsidian Vault ──► File Watcher ──► Indexer ──► SQLite DB      │
│       │                                              │           │
│       │                                              ▼           │
│       │                                         Sync Service     │
│       │                                              │           │
│       │                                              ▼           │
│       ▼                                        PostgreSQL DB     │
│  Inbox Processor                              (via Prisma ORM)   │
│  (Rule-based + LLM classification)                   │           │
│                                                      ▼           │
│  FastAPI Server (port 8000)                  Next.js Frontend    │
│  ├── /health      - Enhanced status check                       │
│  ├── /config      - Provider/model configuration                │
│  ├── /search      - Full-text & semantic search                 │
│  ├── /ask         - Multi-provider RAG Q&A                      │
│  ├── /index       - Document indexing with jobs                 │
│  ├── /inbox/*     - Inbox processing                            │
│  ├── /sync/*      - PostgreSQL sync                             │
│  └── /conversations/* - Chat history                            │
│                                                                  │
│  LLM Providers: OpenAI, Gemini, Anthropic, Ollama               │
│  RAG Techniques: Basic, Hybrid, Rerank, HyDE, Multi-Query       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Completed Features

### 1. Core Indexing System ✅
- **File watcher** monitors vault for changes in real-time
- **Parser** extracts frontmatter, sections, tags, and wikilinks
- **SQLite database** with FTS5 full-text search
- **Incremental indexing** only processes changed files
- **Background job indexing** with progress tracking

### 2. Multi-Provider LLM Support ✅ (NEW)
- **OpenAI** - GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Google Gemini** - Gemini 2.5 Flash/Pro, Gemini 2.0, Gemini 1.5 Pro/Flash
- **Anthropic Claude** - Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3 Opus/Haiku
- **Ollama (Local)** - Any locally installed models
- Runtime provider/model selection via API
- Provider availability checking

### 3. Advanced RAG Techniques ✅ (NEW)
- **Basic RAG** - Simple semantic search and retrieval
- **Hybrid Search** - Combines semantic (embeddings) + keyword (BM25) using RRF
- **Re-ranking** - Cross-encoder model (ms-marco-MiniLM) for improved relevance
- **HyDE** - Hypothetical Document Embeddings for better retrieval
- **Multi-Query** - Generates query variations for comprehensive coverage

### 4. Search Capabilities ✅
- Full-text search across all indexed content (FTS5/BM25)
- **Semantic search** via POST /search endpoint
- Vector embeddings for similarity matching
- Configurable RAG technique for retrieval

### 5. Inbox Processor ✅
- **Location**: `app/inbox_processor.py`
- Processes files in `00_Inbox` folder
- Configurable via YAML config file (`inbox_config.yaml`)
- Classification methods:
  - `rules_only` - Pattern matching only
  - `llm_only` - LLM-based classification
  - `rules_then_llm` - Rules first, LLM as fallback
  - `llm_with_rules_validation` - LLM with rule validation

### 6. Cron Jobs Runner ✅
- **Location**: `cron_jobs.py`
- Commands:
  - `python cron_jobs.py inbox` - Process inbox
  - `python cron_jobs.py reindex` - Full reindex
  - `python cron_jobs.py generate-config` - Create sample config

### 7. PostgreSQL/Prisma Integration ✅
- **Prisma Schema**: `prisma/schema.prisma`
- **PostgreSQL Adapter**: `app/db_postgres.py`
- **Sync Service**: `app/sync_service.py`
- Sync modes: Full, Incremental, Single file
- Conversation management for chat UI

### 8. Frontend API Compatibility ✅ (NEW)
- **CORS enabled** for localhost:3000
- **GET /config** - Returns available providers, models, RAG techniques

### 9. API Security & Public Access ✅ (NEW)
- **API Key Authentication** via `X-API-Key` header
- **Cloudflare Tunnel** for secure public access
- **Public URL**: https://brain.nikolayvalev.com
- Public endpoints: `/health`, `/docs`, `/redoc`, `/openapi.json`
- All other endpoints require authentication
- **Enhanced /health** - Provider status and vector store info
- **POST /index** - Background job indexing with job_id
- **GET /index/status** - Indexing progress tracking

---

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Enhanced health check with provider status |
| `/config` | GET | Available providers, models, RAG techniques |
| `/stats` | GET | Database statistics |
| `/search` | GET | Full-text search (FTS5) |
| `/search` | POST | Semantic search with RAG technique |
| `/ask` | POST | RAG Q&A with provider/model selection |
| `/index` | POST | Trigger background indexing |
| `/index/status` | GET | Indexing progress |
| `/reindex` | POST | Legacy reindex endpoint |
| `/embeddings/stats` | GET | Embedding statistics |
| `/embeddings/generate` | POST | Generate embeddings |
| `/file` | GET | Get file content |
| `/tags` | GET | List all tags |
| `/backlinks` | GET | Find backlinks |
| `/inbox/process` | POST | Process inbox |
| `/inbox/files` | GET | List inbox files |
| `/sync` | POST | Sync to PostgreSQL |
| `/sync/stats` | GET | PostgreSQL stats |
| `/conversations` | POST/GET | Manage conversations |
| `/conversations/{id}` | GET | Get conversation |
| `/conversations/{id}/messages` | POST | Add message |

---

## Configuration

### Environment Variables (.env)
```env
# Required
VAULT_PATH=C:\Users\Nikolay\Code\Shared

# SQLite
DATABASE_PATH=./second_brain.db

# PostgreSQL (for Next.js)
DATABASE_URL=postgresql://user:pass@host:5432/second_brain

# LLM Provider (default)
LLM_PROVIDER=gemini  # openai, gemini, ollama, anthropic

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Google Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Ollama (Local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Reranker (for rerank RAG technique)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# API
API_HOST=127.0.0.1
API_PORT=8000
API_KEY=your-secret-api-key  # Leave empty to disable auth (dev mode)
```

---

## File Structure

```
SecondBrain/
├── app/
│   ├── main.py             # FastAPI application
│   ├── config.py           # Configuration
│   ├── db.py               # SQLite database
│   ├── db_postgres.py      # PostgreSQL adapter
│   ├── sync_service.py     # SQLite → PostgreSQL sync
│   ├── indexer.py          # File indexing
│   ├── parser.py           # Markdown parsing
│   ├── watcher.py          # File system watcher
│   ├── chunker.py          # Text chunking
│   ├── embeddings.py       # Vector embeddings
│   ├── vector_search.py    # Semantic search
│   ├── rag.py              # Legacy RAG service
│   ├── rag_techniques.py   # RAG technique implementations (NEW)
│   ├── llm.py              # Multi-provider LLM service
│   └── inbox_processor.py  # Inbox processing
├── prisma/
│   ├── schema.prisma       # Prisma schema
│   └── migrations/
│       └── init.sql        # PostgreSQL migration
├── tests/
│   ├── test_*.py           # All test files
│   └── conftest.py         # Test fixtures
├── docs/
│   ├── API_REFERENCE.md    # Complete API docs
│   ├── RAG_GUIDE.md
│   ├── LLM_CONFIGURATION.md
│   ├── PRISMA_INTEGRATION.md
│   └── PROJECT_PROGRESS.md # This file
├── cron_jobs.py            # CLI for scheduled tasks
├── run_migration.py        # PostgreSQL setup script
├── inbox_config.yaml       # Inbox configuration
├── requirements.txt
└── .env
```

---

## Running the Service

### Start Server
```bash
cd SecondBrain
./venv/Scripts/Activate.ps1  # Windows
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Example API Calls

All protected endpoints require the `X-API-Key` header:

```bash
# Health check (public - no auth needed)
curl https://brain.nikolayvalev.com/health

# Get available configuration (requires API key)
curl -H "X-API-Key: $API_KEY" https://brain.nikolayvalev.com/config

# Ask with specific provider
curl -X POST https://brain.nikolayvalev.com/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "What are my project ideas?",
    "provider": "gemini",
    "rag_technique": "hybrid"
  }'

# Semantic search
curl -X POST https://brain.nikolayvalev.com/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "machine learning", "limit": 5}'

# Trigger indexing
curl -X POST https://brain.nikolayvalev.com/index \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"force": true}'
```

---

## Dependencies

### Python (requirements.txt)
- fastapi, uvicorn
- watchdog
- python-dotenv, PyYAML
- openai, google-genai, anthropic, httpx
- asyncpg (PostgreSQL)
- sentence-transformers (for reranking)
- pytest, pytest-asyncio

### Next.js (package.json)
- @prisma/client
- prisma

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=app
```

---

## Changelog

### 2026-02-08 (Current)
- **Added API Key Authentication** via `X-API-Key` header middleware
- **Added Cloudflare Tunnel** for secure public access at https://brain.nikolayvalev.com
- Public endpoints: `/health`, `/docs`, `/redoc`, `/openapi.json` (no auth required)
- All other endpoints require valid API key
- API key configurable via `API_KEY` environment variable

### 2026-02-01
- **Added multi-provider LLM support** (OpenAI, Gemini, Anthropic, Ollama)
- **Added 5 RAG techniques** (Basic, Hybrid, Rerank, HyDE, Multi-Query)
- **Added GET /config endpoint** for frontend configuration
- **Enhanced /health endpoint** with provider status
- **Added POST /search** for semantic search
- **Added POST /index** with background job support
- **Added GET /index/status** for progress tracking
- **Added CORS middleware** for frontend integration
- Updated API to match BACKEND_API_SPEC.md

### 2026-01-31
- Migrated from deprecated `google-generativeai` to `google-genai` SDK
- Added inbox processor with configurable rules
- Added cron job runner
- Added PostgreSQL/Prisma integration
- Added sync service for SQLite → PostgreSQL
- Added conversation endpoints for chat UI
- Fixed watcher deadlock bug
- Created comprehensive documentation

---

## Next Steps

### For Next.js Frontend
1. Point `PYTHON_API_URL` to `http://localhost:8000`
2. Fetch `/config` on load to get available providers/models
3. Pass user selections to `/ask` endpoint
4. Use `/search` for direct semantic search

### Future Enhancements
- [x] API Key authentication (completed 2026-02-08)
- [x] Public access via Cloudflare Tunnel (completed 2026-02-08)
- [ ] Streaming responses (SSE) for /ask
- [ ] Token usage tracking
- [ ] Custom system prompts per conversation
- [ ] Real-time WebSocket updates
- [ ] Graph visualization of note connections
- [ ] Model-specific parameter tuning

---

## Contact

For issues or questions, refer to the documentation in `/docs` or check the test files for usage examples.
