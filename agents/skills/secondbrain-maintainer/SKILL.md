---
name: secondbrain-maintainer
description: Maintain and extend the SecondBrain codebase with low-risk, test-validated edits. Use when work touches FastAPI route/model/service layers, indexing/parsing, SQLite FTS, embeddings/RAG techniques, PostgreSQL sync, conversations, cron jobs, migrations, or tests.
---

# SecondBrain Maintainer

Use this workflow for all repository code changes.

## Workflow

1. Read `references/architecture.md`.
2. Read relevant ADRs in `docs/adr/` when the change is cross-cutting.
3. Map the request to modules before editing.
4. Change the smallest surface that fully solves the task.
5. Run targeted tests first, then wider tests when risk is high.
6. Report behavior changes, touched files, and remaining risks.
7. Trigger `agent-retrospective` after completion.

## Module Mapping

- App composition/lifecycle: `app/main.py`
- API routes (thin handlers only): `app/api/routes/*.py`
- API contracts: `app/api/models/*.py`
- Domain logic: `app/services/*.py`
- Core indexing/search store: `app/parser.py`, `app/indexer.py`, `app/db.py`, `app/vector_search.py`
- RAG pipeline: `app/rag_techniques.py`, `app/embeddings.py`, `app/llm.py`, `app/services/rag_service.py`
- Sync and mirror datastore: `app/sync_service.py`, `app/db_postgres.py`, `app/services/sync_api_service.py`, `prisma/schema.prisma`
- Background jobs and automation: `cron_jobs.py`, `agents/scripts/*.ps1`

## Guardrails

- Keep business logic out of route handlers; implement in `app/services/*`.
- Keep schema/model/route changes synchronized.
- Preserve existing singleton patterns unless requested otherwise.
- Keep async boundaries explicit; avoid blocking calls in async paths.
- Keep embedding serialization compatible with existing packed-float format.
- Preserve `EMBEDDING_PROVIDER` override behavior when changing provider logic.
- Preserve API response shapes unless a contract change is requested.
- Add or update tests with every behavioral change.

## Validation

- Route/model/service changes: `pytest tests/test_api.py -v`
- Indexing/search changes: `pytest tests/test_indexer.py tests/test_db.py tests/test_vector_search.py -v`
- RAG/LLM changes: `pytest tests/test_llm.py tests/test_vector_search.py -v`
- Inbox/automation changes: `pytest tests/test_inbox_processor.py tests/test_cron_jobs.py -v`
- Large refactors: `pytest`
