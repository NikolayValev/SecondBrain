# Frontend Implementation Guide (Current)

This is the authoritative frontend integration contract for the current backend.

## API Contract

### Base URL

- Local: `http://127.0.0.1:8000`
- Hosted: `https://brain.nikolayvalev.com`
- Configure frontend with `PYTHON_API_URL`.

### Auth Method

- Protected endpoints require `X-API-Key`.
- Send API key from server-side frontend code only.
- Do not expose `BRAIN_API_KEY` in browser bundles.

### Public Endpoints

- Always public:
  - `GET /health`
- Conditionally public (config-dependent):
  - `GET /config` when `EXPOSE_CONFIG_PUBLIC=true`
  - `GET /docs`, `GET /redoc`, `GET /openapi.json` when `EXPOSE_API_DOCS=true`

### Endpoints Used by UI

| Area | Method | Path | Auth | Notes |
|---|---|---|---|---|
| Boot | GET | `/health` | No | Service reachability |
| Boot | GET | `/config` | Usually yes | Provider/model/technique options |
| Dashboard | GET | `/stats` | Yes | Index stats |
| Search | GET | `/search` | Yes | Full-text search |
| Search | POST | `/search` | Yes | Semantic search |
| Chat | POST | `/ask` | Yes | Sync or SSE answer |
| Conversations | POST | `/conversations` | Yes | Create conversation |
| Conversations | GET | `/conversations` | Yes | List recent conversations |
| Conversations | GET | `/conversations/{id}` | Yes | Conversation detail |
| Conversations | POST | `/conversations/{id}/messages` | Yes | Add message |
| Files | GET | `/file` | Yes | File + metadata |
| Files | GET | `/tags` | Yes | Tag list |
| Files | GET | `/backlinks` | Yes | Link graph by file |
| Graph | GET | `/graph/links` | Yes | Full graph payload |
| Indexing | POST | `/reindex` | Yes | Sync scan |
| Indexing | POST | `/index` | Yes | Background index job |
| Indexing | GET | `/index/status` | Yes | Job status snapshot |
| Realtime | WS | `/ws/index/status` | Yes | Live indexing status |
| Embeddings | GET | `/embeddings/stats` | Yes | Chunk/embedding stats |
| Embeddings | POST | `/embeddings/generate` | Yes | Generate pending embeddings |
| Inbox | GET | `/inbox/files` | Yes | Flat file list |
| Inbox | GET | `/inbox/contents` | Yes | Recursive tree |
| Inbox | POST | `/inbox/process` | Yes | Classify/tag/move |
| Sync (optional) | POST | `/sync` | Yes | Full/incremental mirror sync |
| Sync (optional) | GET | `/sync/stats` | Yes | PG mirror stats |
| Sync (optional) | POST | `/sync/file` | Yes | Single file sync |
| Sync (optional) | POST | `/sync/conversations` | Yes | Conversation sync |
| Sync (optional) | GET | `/sync/changes` | Yes | Change summary since timestamp |
| Security/Admin | GET | `/security/self-check` | Yes | Runtime security posture |

## Request/Response JSON Examples

### `POST /ask` (success, sync)

Request:

```json
{
  "question": "What are my current priorities?",
  "conversation_id": "42",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "rag_technique": "hybrid",
  "temperature": 0.2,
  "max_tokens": 512,
  "top_p": 0.9,
  "top_k": 40,
  "include_sources": true,
  "stream": false
}
```

Response:

```json
{
  "answer": "Your current priorities are...",
  "sources": [
    {
      "path": "Projects/Plan.md",
      "title": "Plan",
      "snippet": "Top priorities this week...",
      "score": 0.932
    }
  ],
  "conversation_id": "42",
  "model_used": "gpt-4o-mini",
  "tokens_used": {
    "prompt": 812,
    "completion": 167,
    "total": 979
  }
}
```

Error example (`400`):

```json
{
  "error": "Unsupported LLM provider: foo",
  "code": "RAG_ERROR"
}
```

### `POST /search` (semantic success)

Request:

```json
{
  "query": "oauth middleware",
  "limit": 10,
  "rag_technique": "hybrid"
}
```

Response:

```json
{
  "results": [
    {
      "path": "Security/API.md",
      "title": "API Security",
      "snippet": "Use API key middleware...",
      "score": 0.8872,
      "metadata": {
        "tags": ["security"],
        "created_at": "2026-02-20T10:00:00",
        "updated_at": "2026-02-25T09:00:00"
      }
    }
  ],
  "query_embedding_time_ms": 35.2,
  "search_time_ms": 35.2
}
```

Error example (`400`):

```json
{
  "detail": "Query cannot be empty"
}
```

### `GET /conversations?session_id=web-user-1&limit=20`

Response:

```json
{
  "conversations": [
    {
      "id": 42,
      "session_id": "web-user-1",
      "title": "Architecture Q&A",
      "created_at": "2026-02-26T20:00:00",
      "updated_at": "2026-02-26T20:15:00"
    }
  ],
  "count": 1
}
```

### `GET /graph/links?max_edges=2000&include_dangling=true`

Response:

```json
{
  "nodes": [
    {
      "id": "Notes/A.md",
      "label": "A",
      "path": "Notes/A.md",
      "node_type": "note",
      "tags": ["project"],
      "in_degree": 1,
      "out_degree": 2,
      "degree": 3
    }
  ],
  "edges": [
    {
      "source": "Notes/A.md",
      "target": "Notes/B.md",
      "label": "B",
      "resolved": true
    }
  ],
  "total_nodes": 2,
  "total_edges": 1,
  "resolved_edges": 1,
  "dangling_edges": 0
}
```

### `GET /index/status`

Response:

```json
{
  "status": "indexing",
  "documents_indexed": 124,
  "documents_pending": 87,
  "last_indexed_at": "2026-02-26T21:30:00",
  "current_job": {
    "job_id": "1f8c2a13-...",
    "progress": 0.42,
    "documents_processed": 21,
    "documents_total": 50
  }
}
```

### `GET /inbox/contents`

Response:

```json
{
  "inbox_path": "00_Inbox",
  "total_files": 3,
  "total_folders": 1,
  "root_files": [
    {
      "name": "capture.md",
      "path": "00_Inbox/capture.md",
      "size_bytes": 482,
      "modified": "2026-02-26T18:00:00",
      "type": "file"
    }
  ],
  "folders": []
}
```

### Standard auth/rate-limit errors

`401`:

```json
{
  "detail": "Missing API key. Include 'X-API-Key' header."
}
```

`403`:

```json
{
  "detail": "Invalid API key"
}
```

`429`:

```json
{
  "detail": "Rate limit exceeded"
}
```

## Data Model Changes (Frontend Impact)

### Added Fields

- `POST /ask` request:
  - `system_prompt`
  - `stream`
  - `temperature`, `max_tokens`, `top_p`, `top_k`
- `POST /ask` response:
  - `conversation_id`
  - `model_used`
  - `tokens_used` (optional)
- `POST /conversations` request:
  - `system_prompt`
- New graph payload:
  - `/graph/links` (`nodes`, `edges`, degree metrics)

### Added Endpoints

- `GET /graph/links`
- `WS /ws/index/status`
- `GET /security/self-check`

### Field Shape Differences to Handle

- Full-text search uses `file_path`; semantic search uses `path`.
- `conversation_id` in `/ask` request accepts string form; backend resolves numeric ids and legacy session-like ids.

### Required Fields and Enums

- `POST /ask`:
  - required: `question`
  - expected provider values: `openai`, `gemini`, `ollama`, `anthropic`
  - `rag_technique` values from `/config` (currently `basic`, `hybrid`, `rerank`, `hyde`, `multi-query`)
- `POST /sync`:
  - `mode`: `full` or `incremental`
- Graph node:
  - `node_type`: `note` or `dangling`

## Behavior Changes

### Pagination / Filter / Sort

- `GET /search`:
  - `limit` only (`1..100`)
- `POST /search`:
  - `limit` (default 10)
- `GET /conversations`:
  - filter: `session_id`
  - limit: `1..100` (default 20)
- `GET /graph/links`:
  - `max_edges` (`1..20000`)
  - `include_dangling` (bool)
- `GET /sync/changes`:
  - `since` ISO-8601 timestamp
- `GET /tags` sorting:
  - by `file_count` desc, then `name` asc

### Validation Rules

- `/search` GET: `q` min length 1, `limit` `1..100`
- `/ask`:
  - `temperature` `0.0..2.0`
  - `max_tokens` `1..32768`
  - `top_p` `(0.0..1.0]`
  - `top_k` `1..500`
- `/embeddings/generate`: `limit` `1..1000`
- `/graph/links`: `max_edges` `1..20000`
- `/sync/changes`: `since` must be ISO-8601
- `/file`: path traversal and absolute paths are rejected (`400`)

### Error Codes and Messages

- `400`: invalid input (for example, empty query, invalid sync mode, invalid `since`)
- `401`: missing API key
- `403`: invalid API key
- `404`: not found (file, conversation, etc.)
- `413`: request too large
- `429`: rate limit exceeded
- `500`: internal error
- `503`: dependency/config unavailable (for example PostgreSQL not configured)

## Realtime and Background

### SSE (`POST /ask` with `stream=true`)

- Event format: `data: <json>\n\n`
- Event types:
  - `source`: initial source metadata
  - `token`: streamed answer token chunks
  - `done`: terminal event with `conversation_id` and `model_used`

### WebSocket (`WS /ws/index/status`)

- Auth when key is required:
  - `X-API-Key` header, or `api_key` query parameter
- Optional query:
  - `interval_ms` clamped to `100..10000`
- Message payload: same shape as `GET /index/status`

### Job/Processing States UI Should Represent

- Indexing:
  - `idle`
  - `indexing`
- Inbox processing response counters:
  - `processed`, `moved`, `skipped`, `errors`
- Sync response:
  - `status`: `completed` or `completed_with_errors`

## Migration Constraints

### Must Stay Compatible

- Keep both search modes in UI:
  - `GET /search` (full-text)
  - `POST /search` (semantic)
- Keep existing conversation continuity behavior:
  - persist and send `conversation_id` returned by `/ask`
- Keep support for both sync and streaming `/ask` responses.

### Can Be Removed

- Assumption that `/config` is always public.
- Direct browser calls to protected backend endpoints with embedded API key.
- Legacy UI paths that ignore `conversation_id` and start a new thread every ask.

## Frontend Acceptance Checklist

- [ ] API proxy layer injects `X-API-Key` server-side.
- [ ] Boot sequence handles `/health`, `/config`, `/stats`.
- [ ] Search UI supports both full-text and semantic routes.
- [ ] Chat supports sync + SSE and preserves conversation continuity.
- [ ] Conversation list/detail/send wired to current contracts.
- [ ] Graph UI supports dangling nodes and degree metrics.
- [ ] Indexing UI supports `/reindex`, `/index`, `/index/status`, and optional websocket stream.
- [ ] Inbox UI supports list/tree/process endpoints with auth.
- [ ] Optional sync UI supports `/sync*` contracts and `completed_with_errors`.
- [ ] UI error handling covers 400/401/403/404/413/429/500/503.
