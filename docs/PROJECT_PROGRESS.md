# Second Brain Project Progress

## Overview

Second Brain is a Python-powered knowledge management system that indexes an Obsidian vault and provides intelligent search, RAG-powered Q&A, and automated document processing. The system now supports PostgreSQL sync for Next.js frontend integration.

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
│       │                                        PostgreSQL DB     │
│       │                                              │           │
│       ▼                                              ▼           │
│  Inbox Processor                              Next.js Frontend   │
│  (Rule-based + LLM classification)            (via Prisma ORM)   │
│                                                                  │
│  FastAPI Server (port 8000)                                      │
│  ├── /search      - Full-text search                            │
│  ├── /ask         - RAG-powered Q&A                             │
│  ├── /inbox/*     - Inbox processing                            │
│  ├── /sync/*      - PostgreSQL sync                             │
│  └── /conversations/* - Chat history                            │
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

### 2. Search & RAG ✅
- Full-text search across all indexed content
- Vector embeddings for semantic search
- RAG-powered Q&A using LLM providers:
  - OpenAI (GPT-4)
  - Google Gemini (migrated to new `google-genai` SDK)
  - Ollama (local models)

### 3. Inbox Processor ✅
- **Location**: `app/inbox_processor.py`
- Processes files in `00_Inbox` folder
- Configurable via YAML config file (`inbox_config.yaml`)
- Classification methods:
  - `rules_only` - Pattern matching only
  - `llm_only` - LLM-based classification
  - `rules_then_llm` - Rules first, LLM as fallback
  - `llm_with_rules_validation` - LLM with rule validation
- Features:
  - Regex pattern matching for file routing
  - Tag-based classification
  - Frontmatter matching
  - LLM-powered smart classification
  - Conflict resolution (skip/overwrite/rename/merge)
  - Metadata preservation

### 4. Cron Jobs Runner ✅
- **Location**: `cron_jobs.py`
- Commands:
  - `python cron_jobs.py inbox` - Process inbox
  - `python cron_jobs.py reindex` - Full reindex
  - `python cron_jobs.py generate-config` - Create sample config

### 5. PostgreSQL/Prisma Integration ✅
- **Prisma Schema**: `prisma/schema.prisma`
- **PostgreSQL Adapter**: `app/db_postgres.py`
- **Sync Service**: `app/sync_service.py`
- Sync modes:
  - Full sync (clears and rebuilds)
  - Incremental sync (changes only)
  - Single file sync
- API endpoints:
  - `POST /sync` - Trigger sync
  - `GET /sync/stats` - Database statistics
  - `POST /sync/file` - Sync single file
- Conversation management for chat UI:
  - `POST /conversations` - Create conversation
  - `GET /conversations/{id}` - Get with messages
  - `POST /conversations/{id}/messages` - Add message

---

## Current Database Status

### SQLite (Primary)
- **Path**: `second_brain.db`
- **Files indexed**: 85
- **Sections**: 393
- **Tags**: 60+
- **Links**: 25

### PostgreSQL (For Next.js)
- **Status**: Connected and synced
- **Last sync**: 2026-01-31
- **Tables**: 13 (files, sections, tags, chunks, embeddings, conversations, messages, etc.)

---

## API Reference

### Health & Stats
```http
GET /health
GET /stats
```

### Search
```http
GET /search?q=your+query&limit=20
POST /ask
{
    "question": "What are my notes about...",
    "include_sources": true
}
```

### Indexing
```http
POST /index/full
POST /index/incremental
POST /embeddings/generate?limit=100
```

### Inbox
```http
POST /inbox/process
{"dry_run": false}

GET /inbox/files
```

### PostgreSQL Sync
```http
POST /sync
{"mode": "full"}  // or "incremental"

GET /sync/stats

POST /sync/file?path=notes/my-note.md
```

### Conversations
```http
POST /conversations
{"session_id": "user-123", "title": "Research"}

GET /conversations?session_id=user-123

GET /conversations/{id}

POST /conversations/{id}/messages
{"role": "user", "content": "..."}
```

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
POSTGRES_SYNC_ENABLED=true

# LLM Provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-1.5-pro

# API
API_HOST=127.0.0.1
API_PORT=8000
```

### Inbox Config (inbox_config.yaml)
```yaml
inbox_folder: "00_Inbox"
classification_method: "rules_then_llm"
conflict_resolution: "rename"

rules:
  - name: "Meeting Notes"
    destination_folder: "meetings"
    patterns: ["^meeting", "^standup"]
    add_tags: ["meeting"]
    
  - name: "Projects"
    destination_folder: "projects"
    required_tags: ["project"]
```

---

## File Structure

```
SecondBrain/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration
│   ├── db.py             # SQLite database
│   ├── db_postgres.py    # PostgreSQL adapter
│   ├── sync_service.py   # SQLite → PostgreSQL sync
│   ├── indexer.py        # File indexing
│   ├── parser.py         # Markdown parsing
│   ├── watcher.py        # File system watcher
│   ├── chunker.py        # Text chunking
│   ├── embeddings.py     # Vector embeddings
│   ├── vector_search.py  # Semantic search
│   ├── rag.py            # RAG service
│   ├── llm.py            # LLM providers
│   └── inbox_processor.py # Inbox processing
├── prisma/
│   ├── schema.prisma     # Prisma schema
│   └── migrations/
│       └── init.sql      # PostgreSQL migration
├── tests/
│   ├── test_*.py         # All test files
│   └── conftest.py       # Test fixtures
├── docs/
│   ├── API_REFERENCE.md
│   ├── RAG_GUIDE.md
│   ├── LLM_CONFIGURATION.md
│   └── PRISMA_INTEGRATION.md
├── cron_jobs.py          # CLI for scheduled tasks
├── run_migration.py      # PostgreSQL setup script
├── inbox_config.yaml     # Inbox configuration
├── requirements.txt
└── .env
```

---

## Running the Service

### Start Server
```bash
cd SecondBrain
./venv/Scripts/Activate.ps1  # Windows
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Process Inbox
```bash
python cron_jobs.py inbox --config inbox_config.yaml
```

### Sync to PostgreSQL
```bash
curl -X POST http://localhost:8000/sync -d '{"mode": "full"}'
```

---

## Next Steps

### For Next.js Frontend
1. Install Node.js
2. Create Next.js app: `npx create-next-app@latest second-brain-ui`
3. Install Prisma: `npm install prisma @prisma/client`
4. Copy `prisma/schema.prisma` to the Next.js project
5. Run `npx prisma generate`
6. Implement API routes (see `docs/PRISMA_INTEGRATION.md`)

### Future Enhancements
- [ ] Real-time WebSocket updates
- [ ] Graph visualization of note connections
- [ ] Daily note templates
- [ ] Spaced repetition for flashcards
- [ ] Mobile app integration

---

## Dependencies

### Python (requirements.txt)
- fastapi, uvicorn
- watchdog
- python-dotenv, PyYAML
- openai, google-genai, httpx
- asyncpg (PostgreSQL)
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
python -m pytest tests/test_inbox_processor.py -v

# Run with coverage
python -m pytest tests/ --cov=app
```

**Test Status**: All 100+ tests passing

---

## Changelog

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

## Contact

For issues or questions, refer to the documentation in `/docs` or check the test files for usage examples.
