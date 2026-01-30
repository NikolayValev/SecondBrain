# API Reference

Complete API documentation for Second Brain.

## Base URL

```
http://127.0.0.1:8000
```

Interactive documentation available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## System Endpoints

### Health Check

Check if the daemon is running.

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "vault_path": "C:\\Users\\Name\\Documents\\Vault",
  "watcher_running": true
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

### Reindex

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

## Search Endpoints

### Full-Text Search

Search across all indexed content using FTS5.

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

**Notes:**
- Snippets include `<mark>` tags for highlighting
- Lower rank values indicate better matches (BM25)

---

## File Endpoints

### Get File Content

Retrieve the full content of a file.

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
  "content": "# Daily Note\n\n## Tasks\n- [x] Review code..."
}
```

**Errors:**

| Code | Description |
|------|-------------|
| 404 | File not found in index |

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
    {"name": "idea", "file_count": 18},
    {"name": "todo", "file_count": 12}
  ],
  "count": 3
}
```

---

### Get Backlinks

Find files that link to a specific file.

```http
GET /backlinks?path=<relative-path>
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `path` | string | Target file path or name |

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

## RAG Endpoints

### Ask Question

Ask a question using RAG (Retrieval-Augmented Generation).

```http
POST /ask
Content-Type: application/json
```

**Request Body:**

```json
{
  "question": "What are my notes about machine learning?",
  "include_sources": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | (required) | The question to answer |
| `include_sources` | boolean | `true` | Include source references |

**Response:**

```json
{
  "answer": "Based on your notes, you have several entries about machine learning. Your 'ML Basics' note covers supervised and unsupervised learning...",
  "sources": [
    {
      "file_path": "notes/ml-basics.md",
      "file_title": "ML Basics",
      "section": "Introduction",
      "similarity": 0.847
    },
    {
      "file_path": "notes/projects/classifier.md",
      "file_title": "Image Classifier",
      "section": "Model Training",
      "similarity": 0.792
    }
  ],
  "query": "What are my notes about machine learning?"
}
```

**Errors:**

| Code | Description |
|------|-------------|
| 400 | Empty question |
| 503 | LLM not configured (missing API key) |
| 500 | Generation failed |

---

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

| Field | Description |
|-------|-------------|
| `chunk_count` | Total text chunks in database |
| `embedding_count` | Chunks with embeddings |
| `files_with_embeddings` | Files that have been embedded |
| `pending_chunks` | Chunks awaiting embedding |

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

**Errors:**

| Code | Description |
|------|-------------|
| 503 | LLM not configured |
| 500 | Embedding generation failed |

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 500 | Internal server error |
| 503 | Service unavailable (LLM not configured) |

---

## Rate Limits

No rate limits are enforced by default. The server runs locally.

For LLM providers:
- **OpenAI**: Subject to OpenAI rate limits
- **Gemini**: Subject to Google rate limits
- **Ollama**: No rate limits (local)

---

## Examples

### cURL

```bash
# Search
curl "http://127.0.0.1:8000/search?q=python"

# Ask question
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What projects am I working on?"}'

# Generate embeddings
curl -X POST "http://127.0.0.1:8000/embeddings/generate?limit=500"
```

### Python

```python
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000")

# Search
results = client.get("/search", params={"q": "python"}).json()

# Ask question
answer = client.post("/ask", json={
    "question": "What are my main project ideas?"
}).json()

print(answer["answer"])
for source in answer["sources"]:
    print(f"  - {source['file_title']} ({source['similarity']:.2f})")
```

### JavaScript

```javascript
// Search
const results = await fetch('http://127.0.0.1:8000/search?q=python')
  .then(r => r.json());

// Ask question
const answer = await fetch('http://127.0.0.1:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'What projects am I working on?' })
}).then(r => r.json());

console.log(answer.answer);
```
