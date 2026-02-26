---
name: agent-retrospective
description: Capture structured run outcomes and synthesize reusable lessons for continuous improvement. Use after meaningful coding tasks, bug fixes, failed attempts, refactors, or test/debug cycles.
---

# Agent Retrospective

Use this workflow to turn each run into better future behavior.

## Workflow

1. Capture task outcome using `scripts/log_run.py`.
2. Include at least one reusable lesson with `--lesson`.
3. Tag domain/risk areas with `--tag` (api, db, rag, tests, infra, etc).
4. Rebuild synthesized lessons with `scripts/synthesize_lessons.py`.
5. Read `agents/memory/LESSONS.md` before the next major task.
6. Inspect trend metrics with `scripts/dashboard.py`.

## Command Pattern

```powershell
python agents/skills/agent-retrospective/scripts/log_run.py `
  --agent codex `
  --task "Fix failing search test" `
  --status success `
  --summary "Adjusted parser edge case and updated tests" `
  --lesson "Run parser-focused tests before full suite on parsing changes" `
  --tag parser `
  --tag tests `
  --file app/parser.py `
  --file tests/test_parser.py

python agents/skills/agent-retrospective/scripts/synthesize_lessons.py
python agents/skills/agent-retrospective/scripts/dashboard.py --window 100 --days 14 --top 5
```

## Quality Rules

- Record factual outcomes, not opinions.
- Keep lessons concrete and action-oriented.
- Log failures and partial results, not only successes.
- Promote repeated lessons into shared instructions when they recur.
- Create/update ADRs in `docs/adr/` when recurring issues are cross-cutting.
- If behavior/config/ops changed, ensure human docs updates are called out (or explicitly logged as a gap).

Use `references/rubric.md` to classify failures and improvement actions.
