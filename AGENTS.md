# SecondBrain Agent System

This repository contains a shared instruction and memory system for Codex, GitHub Copilot, and Antigravity.

## Operating Order

1. Read this file.
2. Read `agents/memory/LESSONS.md` if it exists.
3. Read ADR index: `docs/adr/README.md` (when present).
4. Load only the needed skill from `agents/skills/`.
5. Execute the task with minimal, validated edits.
6. Log the run and refresh lessons.

## Current Architecture Snapshot

- `app/main.py` is composition + lifespan only; it wires middleware and routers.
- Route handlers live in `app/api/routes/*` and should stay thin.
- Request/response contracts live in `app/api/models/*`.
- Domain logic lives in `app/services/*`.
- SQLite (`app/db.py`) is the primary runtime datastore (files, FTS5, chunks, embeddings, conversations).
- PostgreSQL (`app/db_postgres.py`) is an optional mirror for frontend/Prisma use via `app/sync_service.py` and `app/services/sync_api_service.py`.
- RAG retrieval strategies live in `app/rag_techniques.py`; API orchestration lives in `app/services/rag_service.py`.

## Change Checklist For Agents

1. If changing endpoint behavior, update route + model + service together.
2. If changing retrieval logic, update both `/ask` and semantic `/search` behavior expectations.
3. If changing sync logic, preserve best-effort behavior and avoid blocking local SQLite operations.
4. If changing providers, account for `LLM_PROVIDER` and `EMBEDDING_PROVIDER` interplay.
5. Add or update focused tests for each behavior change.

## ADR Discipline

- Create an ADR in `docs/adr/` when introducing or changing cross-cutting behavior (API boundaries, storage authority, sync semantics, auth model, retrieval policy).
- Keep ADRs short and decision-focused (Context, Decision, Consequences).

## Local Skills

- `secondbrain-maintainer`: Project-specific coding workflow for API, indexing, sync, and RAG changes.
  - Path: `agents/skills/secondbrain-maintainer/SKILL.md`
- `agent-retrospective`: Post-run logging and lesson synthesis for continuous improvement.
  - Path: `agents/skills/agent-retrospective/SKILL.md`

## Continuous Improvement Loop

Run these commands after meaningful work:

```powershell
python agents/skills/agent-retrospective/scripts/log_run.py `
  --agent codex `
  --task "<task>" `
  --status success `
  --summary "<what changed and result>" `
  --lesson "<one reusable lesson>"

python agents/skills/agent-retrospective/scripts/synthesize_lessons.py
```

Use `--agent copilot` or `--agent antigravity` for those agents.

Or use the one-shot wrapper:

```powershell
.\agents\scripts\close_loop.ps1 -Agent codex -Task "<task>" -Status success -Summary "<result>" -Lesson "<lesson>"
```

Install the post-commit auto-log hook once per clone:

```powershell
.\agents\scripts\install_git_hooks.ps1 -Agent codex
```

Tune hook behavior:

```powershell
git config secondbrain.retrospectiveAgent copilot
git config secondbrain.postCommitLog false
```

View trend metrics:

```powershell
.\agents\scripts\show_dashboard.ps1 -Window 100 -Days 14 -Top 5
```

## Guardrails

- Prefer small diffs and targeted tests.
- Keep changes aligned with existing architecture patterns.
- Do not add new frameworks when an existing module already solves the problem.
- Treat `agents/memory/LESSONS.md` as process guidance, not product behavior.
