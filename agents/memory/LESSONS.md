# Agent Lessons
_Auto-generated from `agents/memory/runs.jsonl` on 2026-02-26T21:52:39Z._

## Snapshot
- Runs analyzed: 23
- Success: 23
- Partial: 0
- Failed: 0

## Common Tags
- git (3)
- automation (3)
- tooling (2)
- skills (1)
- automation,git,dashboard (1)
- architecture,agents,adr (1)
- docs,api,rag,llm (1)
- security,api,middleware (1)

## Repeated Lessons
- None yet.

## Repeated Failure Patterns
- None yet.

## Suggested Process Updates
- Attach at least one test-focused tag and run command in each run log.

## Recent Runs
| Timestamp (UTC) | Agent | Status | Summary |
|---|---|---|---|
| 2026-02-25T20:04:31Z | codex | success | Updated README and PROJECT_PROGRESS to match current providers, endpoint visibility rules, active routes, env var nam... |
| 2026-02-25T20:20:36Z | codex | success | Added system_prompt support to /ask and /conversations create, persisted prompt as system message, applied prompt in ... |
| 2026-02-25T20:26:29Z | codex | success | Added authenticated WS /ws/index/status endpoint with interval control, added websocket tests, and updated API/projec... |
| 2026-02-25T22:57:15Z | codex | success | Added GET /graph/links endpoint with graph models/service, router wiring, auth-covered tests, and docs updates; marke... |
| 2026-02-25T23:09:37Z | codex | success | Added /ask tuning fields (temperature/max_tokens/top_p/top_k), threaded through rag_service (sync+stream), added prov... |
| 2026-02-25T23:10:25Z | codex | success | Implemented per-request /ask tuning fields (temperature/max_tokens/top_p/top_k), passed them through RAG sync+stream ... |
| 2026-02-25T23:32:01Z | codex | success | Simplified README and PROJECT_PROGRESS, added docs index, removed duplicated/noisy content, and executed full reindex... |
| 2026-02-26T21:47:07Z | codex | success | Added a concise AGENTS.md section defining when to ask for clarification and strict rules for durable memory updates. |
| 2026-02-26T21:49:29Z | codex | success | Condensed AGENTS.md to non-obvious operating rules and trimmed PROJECT_PROGRESS.md to status/backlog plus links. |
| 2026-02-26T21:52:36Z | codex | success | Labeled README/docs index as human docs, added agent-skip defaults, and encoded human-doc update requirements in AGEN... |
