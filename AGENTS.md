# SecondBrain Agent System

This file is for non-obvious, high-signal guidance only.

## Operating Order

1. Read this file.
2. Read `agents/memory/LESSONS.md` if it exists.
3. Read ADR index: `docs/adr/README.md` (when present).
4. Load only the needed skill from `agents/skills/`.
5. Execute the task with minimal, validated edits.
6. Log the run and refresh lessons.

## Lean Policy

- Keep this file short (target well under 200 lines).
- Keep active skills small and focused (typically 2-3); avoid broad "everything" skills.
- Do not document facts agents can derive quickly via code search/import tracing.
- Document only tribal knowledge: hidden constraints, repeated failure points, and required workflows.
- If blocked, surprised, or lacking domain context, stop and ask the user for clarification.

## Documentation Split

- Agent docs (read by default): this file, `agents/memory/LESSONS.md`, and ADR index `docs/adr/README.md`.
- Human docs (skip by default): `README.md` and `docs/*.md` except ADRs.
- Only read/update human docs when the user asks for docs work or when behavior/config/operations changed and humans need updated guidance.

## Change Checklist For Agents

1. If changing endpoint behavior, update route + model + service together.
2. If changing retrieval logic, update both `/ask` and semantic `/search` behavior expectations.
3. If changing sync logic, preserve best-effort behavior and avoid blocking local SQLite operations.
4. If changing providers, account for `LLM_PROVIDER` and `EMBEDDING_PROVIDER` interplay.
5. Add or update focused tests for each behavior change.
6. If behavior/config/ops changed, update affected human docs in the same task.

## ADR Discipline

- Create an ADR in `docs/adr/` when introducing or changing cross-cutting behavior (API boundaries, storage authority, sync semantics, auth model, retrieval policy).
- Keep ADRs short and decision-focused (Context, Decision, Consequences).

## Local Skills

- `secondbrain-maintainer` for API/indexing/sync/RAG work.
- `agent-retrospective` for run logging and lesson synthesis.

## Continuous Improvement Loop

- After meaningful work, run `.\agents\scripts\close_loop.ps1` (or the underlying retrospective scripts).

## Guardrails

- Prefer small diffs and targeted tests.
- Keep changes aligned with existing architecture patterns.
- Do not add new frameworks when an existing module already solves the problem.
- Treat `agents/memory/LESSONS.md` as process guidance, not product behavior.

## Memory Updates

- Update this file only when a durable, generalizable repo-working lesson is learned.
- Add one short bullet under the most relevant section.
- Do not add bug-specific or code-searchable notes.
