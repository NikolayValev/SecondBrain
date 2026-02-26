# Second Brain Project Progress

## Status Snapshot (2026-02-25)

- Core indexing and watcher pipeline: complete
- Multi-provider RAG (`openai`, `gemini`, `ollama`, `anthropic`): complete
- Conversation APIs and continuity: complete
- Sync APIs (`/sync`, `/sync/file`, `/sync/conversations`, `/sync/changes`): complete
- Security baseline for public mode: complete
- Real-time indexing status WebSocket: complete
- Graph visualization payload endpoint (`/graph/links`): complete

## Completed Backlog (2026-02-25)

- [x] Fix `/ask` conversation continuity and ID shape
- [x] Correct `/sync/changes` `since` semantics
- [x] Add missing tests for RAG + Sync + Conversations API flows
- [x] Add CI workflow to run tests on push/PR
- [x] Reconcile docs with current implementation and remove stale references
- [x] Implement per-conversation custom system prompts
- [x] Add real-time WebSocket updates for indexing status
- [x] Add graph visualization endpoint
- [x] Add model-specific parameter tuning for `/ask`

## Links

- API behavior: [API_REFERENCE.md](./API_REFERENCE.md)
- Retrieval behavior: [RAG_GUIDE.md](./RAG_GUIDE.md)
- Provider setup: [LLM_CONFIGURATION.md](./LLM_CONFIGURATION.md)
- Cross-cutting decisions: [ADR index](./adr/README.md)
