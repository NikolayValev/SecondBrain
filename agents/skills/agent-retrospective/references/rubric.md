# Retrospective Rubric

## Status Labels

- `success`: Requested behavior delivered and validated.
- `partial`: Some value delivered, but scope or validation incomplete.
- `failed`: Requested behavior not delivered or regressed.

## Recommended Tags

- `api`
- `api-layer`
- `service-layer`
- `parser`
- `indexing`
- `db`
- `pg-mirror`
- `sync`
- `rag`
- `embeddings`
- `tests`
- `infra`
- `tooling`

## Lesson Quality Checklist

- Describe a concrete trigger condition.
- Describe a concrete action.
- Keep it short enough to scan quickly.
- Avoid vague statements like "be careful" or "improve quality".

## Escalation Rules

- If the same failure pattern appears 3+ times, update shared instructions.
- If failures repeat in one module, add a focused regression test.
- If a run ends `partial` or `failed`, include at least one `next_step`.
- If a recurring issue spans multiple modules or layers, create/update an ADR in `docs/adr/`.
