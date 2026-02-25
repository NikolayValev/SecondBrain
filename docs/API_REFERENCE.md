# API Reference

Current API reference for the FastAPI app in `app/main.py`.

## Base URL

- Local: `http://127.0.0.1:8000`
- Docs: `/docs` and `/redoc`

## Authentication

If `BRAIN_API_KEY` is set, protected endpoints require header:

```http
X-API-Key: <your-key>
```

Public endpoints (no key required):

- `GET /health`
- `GET /config` only when `EXPOSE_CONFIG_PUBLIC=true`
- `GET /docs`, `GET /redoc`, `GET /openapi.json` only when `EXPOSE_API_DOCS=true`

If `BRAIN_API_KEY` is empty, auth is effectively disabled.

## Endpoint Index

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service + provider health |
| GET | `/stats` | SQLite indexing stats |
| GET | `/config` | Provider/model/technique options |
| GET | `/search` | Full-text (FTS5) search |
| POST | `/search` | Semantic search |
| GET | `/file` | Get file content + metadata |
| GET | `/tags` | List tags with counts |
| GET | `/backlinks` | Find inbound links |
| POST | `/ask` | RAG question answering (sync or stream) |
| GET | `/embeddings/stats` | Embedding/chunk stats |
| POST | `/embeddings/generate` | Generate embeddings for pending chunks |
| POST | `/reindex` | Immediate full/incremental reindex |
| POST | `/index` | Start background indexing job |
| GET | `/index/status` | Background indexing status |
| POST | `/inbox/process` | Process inbox files |
| GET | `/inbox/files` | Flat inbox file list |
| GET | `/inbox/contents` | Recursive inbox tree |
| POST | `/sync` | SQLite -> PostgreSQL sync |
| GET | `/sync/stats` | PostgreSQL sync stats |
| POST | `/sync/file` | Sync one file to PostgreSQL |
| POST | `/sync/conversations` | Sync conversations/messages |
| GET | `/sync/changes` | Count changes since timestamp |
| POST | `/conversations` | Create conversation |
| GET | `/conversations/{conversation_id}` | Get conversation + messages |
| POST | `/conversations/{conversation_id}/messages` | Add conversation message |
| GET | `/conversations` | List recent conversations |
| GET | `/security/self-check` | Security posture report |

## Core Request/Response Shapes

## `GET /health`

Returns:

- `status`
- `version`
- `vault_path`
- `watcher_running`
- `providers` map (`openai`, `gemini`, `ollama`, `anthropic`)
- `vector_store` info

## `GET /config`

Returns:

- `providers[]` with availability + model list
- `rag_techniques[]` (`basic`, `hybrid`, `rerank`, `hyde`, `multi-query`)
- `defaults` (`provider`, `model`, `rag_technique`)
- `embedding_model`
- `vector_store`

## `GET /search`

Query params:

- `q` (required, min length 1)
- `limit` (default `20`, max `100`)

Response:

- `query`
- `results[]` with `file_path`, `title`, `heading`, `snippet`, `rank`
- `count`

## `POST /search`

Body:

```json
{
  "query": "design notes",
  "limit": 10,
  "rag_technique": "hybrid"
}
```

Response:

- `results[]` with `path`, `title`, `snippet`, `score`, `metadata`
- `query_embedding_time_ms`
- `search_time_ms`

## `GET /file`

Query param:

- `path` (vault-relative path)

Response:

- `path`, `title`, `content`
- `tags[]`
- `created_at`, `modified_at`
- `frontmatter` object

## `POST /ask`

Body:

```json
{
  "question": "What did I write about sync architecture?",
  "conversation_id": null,
  "provider": "openai",
  "model": null,
  "rag_technique": "hybrid",
  "include_sources": true,
  "stream": false
}
```

Non-stream response:

- `answer`
- `sources[]` with `path`, `title`, `snippet`, `score`
- `conversation_id`
- `model_used`
- optional `tokens_used` (`prompt`, `completion`, `total`)

Stream mode (`stream=true`):

- Content type: `text/event-stream`
- SSE events emitted as `data: <json>` with:
  - `{"type":"source", ...}` (optional, one per source)
  - `{"type":"token","content":"..."}`
  - `{"type":"done","conversation_id":"...","model_used":"..."}`

## `POST /index`

Body:

```json
{
  "paths": ["notes/a.md", "notes/b.md"],
  "force": false
}
```

Response:

- `status` (`started`)
- `job_id`
- `documents_queued`

`GET /index/status` reports current in-memory job state.

## `POST /sync`

Body:

```json
{
  "mode": "incremental"
}
```

Modes: `incremental` or `full`.

Returns counts for files, sections, tags, links, chunks, embeddings, conversations, messages, plus `errors[]` and status.

## `GET /conversations`

Query params:

- `session_id` (optional)
- `limit` (default `20`, max `100`)

Returns:

- `conversations[]`
- `count`

## `GET /security/self-check`

Returns a security report including:

- `mode` (`local` or `public`)
- `fail_fast` (whether startup enforces failures)
- `safe` (overall pass/fail)
- `failed_checks`, `warning_checks`
- `checks[]` with per-check status (`pass`, `warn`, `fail`)

## Error Notes

- `400`: bad input (for example unknown sync mode, unknown RAG technique)
- `401`: missing API key (when auth enabled)
- `403`: invalid API key
- `404`: missing file/conversation
- `413`: request body too large
- `429`: rate limit exceeded
- `503`: required backend unavailable (for example PostgreSQL not configured)
- `500`: internal error

Rate-limit response headers include:
- `Retry-After`
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Window`

Request-size rejection (`413`) includes:
- `X-Max-Request-Bytes`
- `X-Request-Bytes`
