# Second Brain Copilot Instructions

Follow this order on every coding task:

1. Read `AGENTS.md`.
2. Read `agents/memory/LESSONS.md`.
3. Use `agents/skills/secondbrain-maintainer/SKILL.md` for code changes.
4. Use `agents/skills/agent-retrospective/SKILL.md` after completion.

## Architecture Map

- App composition/lifespan: `app/main.py`
- HTTP handlers: `app/api/routes/*`
- API contracts: `app/api/models/*`
- Business logic: `app/services/*`
- Primary datastore: SQLite (`app/db.py`, includes FTS5 + conversations)
- Optional mirror datastore: PostgreSQL (`app/db_postgres.py`) via `app/sync_service.py`
- RAG pipeline: `app/rag_techniques.py` -> `app/embeddings.py` -> `app/vector_search.py` -> `app/services/rag_service.py` -> `app/llm.py`

## Coding Patterns

- Prefer existing singleton instances for production paths.
- Instantiate classes directly in tests when dependency injection is needed.
- Keep async flows async end-to-end in API and PostgreSQL paths.
- Preserve packed-float embedding compatibility.
- Keep route handlers thin; place behavior changes in service modules.
- Account for `LLM_PROVIDER` and `EMBEDDING_PROVIDER` as separate knobs.

## Verification Commands

```powershell
pytest
pytest tests/test_api.py -v
pytest -k "test_search" -v
python -m uvicorn app.main:app --reload
```

## Continuous Improvement

After meaningful changes, log the run and refresh shared lessons:

```powershell
python agents/skills/agent-retrospective/scripts/log_run.py `
  --agent copilot `
  --task "<task>" `
  --status success `
  --summary "<result>" `
  --lesson "<reusable lesson>"

python agents/skills/agent-retrospective/scripts/synthesize_lessons.py
```

Shortcut:

```powershell
.\agents\scripts\close_loop.ps1 -Agent copilot -Task "<task>" -Status success -Summary "<result>" -Lesson "<lesson>"
```

Install post-commit auto-log (once per clone):

```powershell
.\agents\scripts\install_git_hooks.ps1 -Agent copilot
```

Open metrics dashboard:

```powershell
.\agents\scripts\show_dashboard.ps1 -Agent copilot -Window 100 -Days 14 -Top 5
```
