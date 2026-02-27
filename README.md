# Second Brain

> Human doc for developers/operators.  
> Agents: skip by default and use `AGENTS.md` unless the task explicitly includes human documentation updates.

Second Brain indexes an Obsidian vault and exposes a FastAPI backend for:

- Full-text and semantic search
- RAG question answering (`/ask`) with provider/model selection
- Conversation history endpoints
- Optional SQLite to PostgreSQL sync for frontend consumers

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and set `VAULT_PATH`.
4. Start the API.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload
```

## Common Operations

Run a full reindex:

```powershell
python cron_jobs.py reindex
```

Run incremental reindex:

```powershell
python cron_jobs.py reindex --incremental
```

Generate pending embeddings:

```powershell
curl -X POST "http://127.0.0.1:8000/embeddings/generate?limit=500"
```

## Security (Public Deployments)

If exposed publicly, use API keys and public-mode guardrails:

- `REQUIRE_API_KEY=true`
- `PUBLIC_API_MODE=true`
- `BRAIN_API_KEY=<strong-random-secret>`
- `EXPOSE_API_DOCS=false`
- `EXPOSE_CONFIG_PUBLIC=false`

Runtime posture endpoint: `GET /security/self-check` (authenticated).

## Documentation

- [Documentation Index](docs/README.md)
- [API Reference](docs/API_REFERENCE.md)
- [Frontend Implementation Guide](docs/FRONTEND_IMPLEMENTATION.md)
- [Project Progress](docs/PROJECT_PROGRESS.md)
- [LLM Configuration](docs/LLM_CONFIGURATION.md)
- [RAG Guide](docs/RAG_GUIDE.md)
- [Prisma Integration](docs/PRISMA_INTEGRATION.md)
- [ADRs](docs/adr/README.md)
