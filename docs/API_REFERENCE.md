# API Reference

Complete API documentation for Second Brain.

## Base URL

**Local:**
```
http://127.0.0.1:8000
```

**Public (via Cloudflare Tunnel):**
```
https://brain.nikolayvalev.com
```

Interactive documentation available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## Authentication

All endpoints except public ones require API key authentication via the `X-API-Key` header.

### Public Endpoints (No Auth Required)
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc documentation
- `GET /openapi.json` - OpenAPI schema

### Protected Endpoints
All other endpoints require the `X-API-Key` header.

### Example Request
```bash
curl -X GET "https://brain.nikolayvalev.com/stats" \
  -H "X-API-Key: your-api-key-here"
```

### Error Responses

**401 Unauthorized** - Missing API key:
```json
{
  "detail": "Missing API key. Include 'X-API-Key' header."
}
```

**403 Forbidden** - Invalid API key:
```json
{
  "detail": "Invalid API key"
}
```

### Configuration

Set the API key in your `.env` file:
```env
BRAIN_API_KEY=your-secret-api-key
```

Leave `BRAIN_API_KEY` empty to disable authentication (development mode only).

---

## System Endpoints

### Health Check

Enhanced health check endpoint with provider status.

```http
GET /health
```

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "vault_path": "C:\\Users\\Name\\Documents\\Vault",
  "watcher_running": true,
  "providers": {
    "ollama": { "available": true, "models_loaded": ["llama3.2"] },
    "openai": { "available": true },
    "gemini": { "available": true },
    "anthropic": { "available": false, "error": "API key not configured" }
  },
  "vector_store": {
    "type": "sqlite",
    "documents_indexed": 1234
  }
}
```

---

### Configuration

Get available providers, models, and RAG techniques.

```http
GET /config
```

**Response:**

```json
{
  "providers": [
    {
      "id": "ollama",
      "name": "Ollama (Local)",
      "available": true,
      "base_url": "http://localhost:11434",
      "models": [
        {
          "id": "llama3.2",
          "name": "Llama 3.2",
          "context_length": 128000,
          "available": true
        }
      ]
    },
    {
      "id": "openai",
      "name": "OpenAI",
      "available": true,
      "models": [
        { "id": "gpt-4o", "name": "GPT-4o", "context_length": 128000 },
        { "id": "gpt-4o-mini", "name": "GPT-4o Mini", "context_length": 128000 }
      ]
    },
    {
      "id": "gemini",
      "name": "Google Gemini",
      "available": true,
      "models": [
        { "id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "context_length": 1000000 },
        { "id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "context_length": 2000000 }
      ]
    },
    {
      "id": "anthropic",
      "name": "Anthropic",
      "available": true,
      "models": [
        { "id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "context_length": 200000 },
        { "id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "context_length": 200000 }
      ]
    }
  ],
  "rag_techniques": [
    { "id": "basic", "name": "Basic RAG", "description": "Simple semantic search and retrieval" },
    { "id": "hybrid", "name": "Hybrid Search", "description": "Combines semantic and keyword search (BM25 + embeddings)" },
    { "id": "rerank", "name": "Re-ranking", "description": "Uses a cross-encoder reranker model to improve results" },
    { "id": "hyde", "name": "HyDE", "description": "Hypothetical Document Embeddings - generates hypothetical answer first" },
    { "id": "multi-query", "name": "Multi-Query", "description": "Generates multiple query variations for better coverage" }
  ],
  "defaults": {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "rag_technique": "basic"
  },
  "embedding_model": "text-embedding-004",
  "vector_store": "sqlite"
}
```

---

### Statistics

Get database statistics.

```http
GET /stats
```

**Response:**

```json
{
  "file_count": 150,
  "section_count": 450,
  "tag_count": 75,
  "link_count": 320,
  "last_indexed": "2026-01-30T14:30:00"
}
```

---

## RAG Endpoints

### Ask Question

Ask a question using RAG (Retrieval-Augmented Generation) with provider/model selection.

```http
POST /ask
Content-Type: application/json
```

**Request Body:**

```json
{
  "question": "What are my notes about machine learning?",
  "conversation_id": "optional-uuid",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "rag_technique": "basic",
  "include_sources": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | (required) | The question to answer |
| `conversation_id` | string | null | Optional conversation ID for context |
| `provider` | string | `"gemini"` | LLM provider (ollama, openai, gemini, anthropic) |
| `model` | string | null | Specific model ID to use |
| `rag_technique` | string | `"basic"` | RAG technique (basic, hybrid, rerank, hyde, multi-query) |
| `include_sources` | boolean | `true` | Include source references |

**Response:**

```json
{
  "answer": "Based on your notes, you have several entries about machine learning...",
  "sources": [
    {
      "path": "notes/ml-basics.md",
      "title": "ML Basics",
      "snippet": "Machine learning is a subset of AI that focuses on...",
      "score": 0.847
    }
  ],
  "conversation_id": "uuid-string",
  "model_used": "gemini-2.5-flash",
  "tokens_used": {
    "prompt": 1500,
    "completion": 250,
    "total": 1750
  }
}
```

**Error Response:**

```json
{
  "error": "Error message",
  "code": "PROVIDER_UNAVAILABLE | MODEL_NOT_FOUND | RAG_ERROR"
}
```

---

### Semantic Search

Direct semantic search without generating an answer.

```http
POST /search
Content-Type: application/json
```

**Request Body:**

```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "rag_technique": "basic"
}
```

**Response:**

```json
{
  "results": [
    {
      "path": "notes/ml-basics.md",
      "title": "ML Basics",
      "snippet": "Machine learning algorithms can be categorized into...",
      "score": 0.8547,
      "metadata": {
        "tags": ["ml", "ai"],
        "created_at": "2026-01-15T10:00:00",
        "updated_at": "2026-01-20T14:30:00"
      }
    }
  ],
  "query_embedding_time_ms": 45.2,
  "search_time_ms": 12.8
}
```

---

### Full-Text Search

Search across all indexed content using FTS5 (keyword matching).

```http
GET /search?q=<query>&limit=20
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `q` | string | (required) | Search query |
| `limit` | integer | `20` | Max results (1-100) |

**Response:**

```json
{
  "query": "python",
  "results": [
    {
      "file_path": "notes/programming/python-tips.md",
      "title": "Python Tips",
      "heading": "Best Practices",
      "snippet": "Use <mark>Python</mark> virtual environments for...",
      "rank": -2.5
    }
  ],
  "count": 1
}
```

---

## Indexing Endpoints

### Trigger Indexing

Start document indexing for the knowledge base.

```http
POST /index
Content-Type: application/json
```

**Request Body:**

```json
{
  "paths": ["notes/new-file.md", "projects/project-a.md"],
  "force": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `paths` | string[] | null | Specific paths to index (null for all) |
| `force` | boolean | `false` | Force re-indexing even if unchanged |

**Response:**

```json
{
  "status": "started",
  "job_id": "uuid-string",
  "documents_queued": 42
}
```

---

### Get Indexing Status

```http
GET /index/status
```

**Response:**

```json
{
  "status": "indexing",
  "documents_indexed": 1234,
  "documents_pending": 50,
  "last_indexed_at": "2026-02-01T12:00:00Z",
  "current_job": {
    "job_id": "uuid-string",
    "progress": 0.75,
    "documents_processed": 30,
    "documents_total": 40
  }
}
```

---

### Manual Reindex (Legacy)

Manually trigger vault reindexing.

```http
POST /reindex?full=false
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `full` | boolean | `false` | If true, rescan all files |

**Response:**

```json
{
  "status": "completed",
  "indexed": 15,
  "errors": 0,
  "type": "incremental"
}
```

---

## Embedding Endpoints

### Embedding Statistics

Get embedding generation statistics.

```http
GET /embeddings/stats
```

**Response:**

```json
{
  "chunk_count": 1500,
  "embedding_count": 1200,
  "files_with_embeddings": 45,
  "pending_chunks": 300
}
```

---

### Generate Embeddings

Generate embeddings for pending chunks.

```http
POST /embeddings/generate?limit=100
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `limit` | integer | `100` | Max chunks to process (1-1000) |

**Response:**

```json
{
  "status": "completed",
  "processed": 100,
  "failed": 0,
  "pending_remaining": 200
}
```

---

## File Endpoints

### Get File Content

Retrieve the full content and metadata of a file.

```http
GET /file?path=<relative-path>
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `path` | string | Relative path from vault root |

**Response:**

```json
{
  "path": "notes/daily/2026-01-30.md",
  "title": "Daily Note - January 30",
  "content": "---\ntags: [daily, journal]\ncreated: 2026-01-30\n---\n\n# Daily Note\n\n## Tasks\n- [x] Review code...",
  "tags": ["daily", "journal"],
  "created_at": "2026-01-30T00:00:00",
  "modified_at": "2026-01-30T18:45:00",
  "frontmatter": {
    "tags": ["daily", "journal"],
    "created": "2026-01-30"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Relative file path |
| `title` | string | Document title |
| `content` | string | Full markdown content including frontmatter |
| `tags` | string[] | List of tags from frontmatter and inline tags |
| `created_at` | string | ISO 8601 creation date (from frontmatter or file mtime) |
| `modified_at` | string | ISO 8601 modification date (from file mtime) |
| `frontmatter` | object | Parsed YAML frontmatter as key-value pairs |

**Error Responses:**

- `404 Not Found` - File does not exist in the vault

---

## Metadata Endpoints

### List Tags

Get all tags with file counts.

```http
GET /tags
```

**Response:**

```json
{
  "tags": [
    {"name": "project", "file_count": 25},
    {"name": "idea", "file_count": 18}
  ],
  "count": 2
}
```

---

### Get Backlinks

Find files that link to a specific file.

```http
GET /backlinks?path=<relative-path>
```

**Response:**

```json
{
  "target": "projects/my-app.md",
  "backlinks": [
    {"path": "daily/2026-01-30.md", "title": "Daily Note"},
    {"path": "index.md", "title": "Project Index"}
  ],
  "count": 2
}
```

---

## Inbox Endpoints

### Process Inbox

Process documents in the inbox folder.

```http
POST /inbox/process
Content-Type: application/json
```

**Request Body:**

```json
{
  "dry_run": false
}
```

---

### List Inbox Files

```http
GET /inbox/files
```

---

## PostgreSQL Sync Endpoints

### Sync to PostgreSQL

```http
POST /sync
Content-Type: application/json
```

**Request Body:**

```json
{
  "mode": "full"
}
```

---

### Sync Stats

```http
GET /sync/stats
```

---

## Conversation Endpoints

### Create Conversation

```http
POST /conversations
Content-Type: application/json
```

**Request Body:**

```json
{
  "session_id": "user-123",
  "title": "Research Chat"
}
```

---

### Get Conversation

```http
GET /conversations/{id}
```

---

### Add Message

```http
POST /conversations/{id}/messages
Content-Type: application/json
```

**Request Body:**

```json
{
  "role": "user",
  "content": "What about..."
}
```

---

### List Conversations

```http
GET /conversations?session_id=user-123&limit=20
```

---

## RAG Techniques

| Technique | Description |
|-----------|-------------|
| `basic` | Simple semantic search - embed query, retrieve top-k similar documents |
| `hybrid` | Combines semantic (embeddings) and keyword search (BM25) using RRF |
| `rerank` | Uses cross-encoder model to re-rank initial results |
| `hyde` | Generates hypothetical answer first, then searches for similar documents |
| `multi-query` | Generates query variations for better coverage |

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `PROVIDER_UNAVAILABLE` | LLM provider not configured or unreachable |
| `MODEL_NOT_FOUND` | Specified model not available |
| `RAG_ERROR` | Error in RAG retrieval or generation |
| `GENERATION_ERROR` | LLM generation failed |

---

## Examples

### cURL

```bash
# Get configuration
curl http://127.0.0.1:8000/config

# Ask with specific provider and technique
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What projects am I working on?",
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "rag_technique": "hybrid"
  }'

# Semantic search
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "limit": 5,
    "rag_technique": "basic"
  }'

# Trigger indexing
curl -X POST "http://127.0.0.1:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Check indexing status
curl http://127.0.0.1:8000/index/status
```

### Python

```python
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000")

# Get available config
config = client.get("/config").json()
print(f"Available providers: {[p['id'] for p in config['providers']]}")
print(f"RAG techniques: {[t['id'] for t in config['rag_techniques']]}")

# Ask with provider selection
answer = client.post("/ask", json={
    "question": "What are my main project ideas?",
    "provider": "gemini",
    "rag_technique": "hybrid"
}).json()

print(answer["answer"])
print(f"Model used: {answer['model_used']}")
for source in answer["sources"]:
    print(f"  - {source['title']} ({source['score']:.2f})")
```

### JavaScript

```javascript
// Get configuration
const config = await fetch('http://127.0.0.1:8000/config')
  .then(r => r.json());

// Ask with options
const answer = await fetch('http://127.0.0.1:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'What projects am I working on?',
    provider: 'openai',
    model: 'gpt-4o',
    rag_technique: 'rerank'
  })
}).then(r => r.json());

console.log(answer.answer);
console.log(`Model: ${answer.model_used}`);
```
