# Antigravity Instructions

Use the same shared workflow as Codex and Copilot.

## Startup

1. Read `AGENTS.md`.
2. Read `agents/memory/LESSONS.md`.
3. Load the relevant skill from `agents/skills/`.
4. For cross-cutting changes, read `docs/adr/README.md`.

## Skills

- `agents/skills/secondbrain-maintainer/SKILL.md`
- `agents/skills/agent-retrospective/SKILL.md`

## Architecture Notes

- Keep endpoints thin in `app/api/routes/*`.
- Put behavior in `app/services/*`.
- Keep API models in `app/api/models/*` aligned with route/service changes.
- Treat SQLite as primary state and PostgreSQL sync as best-effort mirror.

## Completion

After each meaningful task, log the run and synthesize lessons:

```powershell
python agents/skills/agent-retrospective/scripts/log_run.py --agent antigravity --task "<task>" --status success --summary "<summary>" --lesson "<lesson>"
python agents/skills/agent-retrospective/scripts/synthesize_lessons.py
```

Shortcut:

```powershell
.\agents\scripts\close_loop.ps1 -Agent antigravity -Task "<task>" -Status success -Summary "<summary>" -Lesson "<lesson>"
```

Install post-commit auto-log:

```powershell
.\agents\scripts\install_git_hooks.ps1 -Agent antigravity
```

View trend metrics:

```powershell
.\agents\scripts\show_dashboard.ps1 -Agent antigravity -Window 100 -Days 14 -Top 5
```
