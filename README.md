# Second Brain

A local daemon that indexes an Obsidian vault and exposes a REST API for search, file access, and **RAG-powered Q&A**. Designed for Windows, with full LLM integration.

## Features

- **File Watching**: Monitors your vault for changes in real-time
- **Full-Text Search**: FTS5-powered keyword search with BM25 ranking
- **Markdown Parsing**: Extracts titles, headings, tags, and links
- **REST API**: FastAPI-based endpoints for integration
- **Incremental Indexing**: Only re-indexes changed files
- **RAG Q&A**: Ask questions about your notes using AI (OpenAI, Gemini, or Ollama)
- **Vector Search**: Semantic similarity search using embeddings
- **Multi-Provider LLM**: Supports OpenAI, Google Gemini, and local Ollama models

## Requirements

- Python 3.10+
- Windows (also works on macOS/Linux)
- An Obsidian vault (or any folder with markdown files)

## Installation

### 1. Clone or Download

```powershell
cd C:\Users\YourName\Projects
git clone <repo-url> SecondBrain
cd SecondBrain
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file and edit it:

```powershell
copy .env.example .env
notepad .env
```

**Required**: Set `VAULT_PATH` to your Obsidian vault location:

```ini
VAULT_PATH=C:\Users\YourName\Documents\ObsidianVault
```

## Running the Server

### Development Mode

```powershell
# From the project root with venv activated
python -m uvicorn app.main:app --reload
```

### Production Mode

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or run directly:

```powershell
python -m app.main
```

The server will:
1. Validate the vault path
2. Initialize the SQLite database
3. Perform an initial scan (or incremental update)
4. Start watching for file changes
5. Serve the API at `http://127.0.0.1:8000`

## API Endpoints

### Health Check
```
GET /health
```
Returns daemon status and vault path.

### Statistics
```
GET /stats
```
Returns counts of indexed files, sections, tags, and links.

### Full-Text Search
```
GET /search?q=<query>&limit=20
```
Search across all indexed content. Returns snippets with highlighted matches.

**Example:**
```powershell
curl "http://127.0.0.1:8000/search?q=python+programming"
```

### Get File Content
```
GET /file?path=<relative-path>
```
Get the full content of a specific file.

**Example:**
```powershell
curl "http://127.0.0.1:8000/file?path=notes/my-note.md"
```

### List Tags
```
GET /tags
```
List all tags with file counts.

### Find Backlinks
```
GET /backlinks?path=<relative-path>
```
Find all files that link to a specific file.

### Manual Reindex
```
POST /reindex?full=false
```
Trigger a manual reindex. Use `full=true` for complete rescan.

### Ask Question (RAG)
```
POST /ask
Content-Type: application/json

{
  "question": "What are my notes about Python?",
  "include_sources": true
}
```
Ask a question using RAG. Returns an AI-generated answer with source references.

**Example:**
```powershell
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "What did I write about productivity?"}'
```

### Embedding Statistics
```
GET /embeddings/stats
```
Get counts of chunks and embeddings.

### Generate Embeddings
```
POST /embeddings/generate?limit=100
```
Generate embeddings for pending chunks. Run after indexing to enable RAG.

## API Documentation

FastAPI provides interactive documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Project Structure

```
SecondBrain/
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI application & endpoints
│   ├── config.py       # Configuration management
│   ├── db.py           # SQLite database with FTS5
│   ├── parser.py       # Markdown parsing
│   ├── indexer.py      # File indexing logic
│   ├── watcher.py      # File system watcher
│   ├── llm.py          # LLM provider abstraction
│   ├── chunker.py      # Text chunking for embeddings
│   ├── embeddings.py   # Embedding generation & storage
│   ├── vector_search.py # Cosine similarity search
│   └── rag.py          # RAG pipeline orchestration
├── docs/
│   └── LLM_CONFIGURATION.md  # LLM setup guide
├── tests/              # Unit tests
├── .env                # Environment configuration
├── .env.example        # Example configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Database Schema

The SQLite database includes:

- **files**: File metadata (path, mtime, title, content)
- **sections**: Headings and their content
- **tags**: Unique tag names
- **file_tags**: File-tag associations
- **links**: Outbound links from files
- **chunks**: Text chunks for embedding
- **embeddings**: Vector embeddings (stored as BLOB)
- **fts_content**: FTS5 virtual table for full-text search

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_PATH` | (required) | Path to your Obsidian vault |
| `DATABASE_PATH` | `./second_brain.db` | SQLite database location |
| `DEBOUNCE_SECONDS` | `1.0` | Delay before processing file changes |
| `API_HOST` | `127.0.0.1` | API server host |
| `API_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai`, `gemini`, or `ollama` |
| `LLM_TEMPERATURE` | `0.7` | Generation temperature (0.0-1.0) |
| `LLM_MAX_TOKENS` | `4096` | Maximum tokens in response |

#### OpenAI
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

#### Google Gemini
| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google AI API key |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Chat model |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Embedding model |

#### Ollama (Local)
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |

See [docs/LLM_CONFIGURATION.md](docs/LLM_CONFIGURATION.md) for detailed setup instructions.

## Quick Start with RAG

1. **Configure LLM** - Add your API key to `.env`:
   ```ini
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your-key-here
   ```

2. **Start server** - Index your vault:
   ```powershell
   python -m uvicorn app.main:app --reload
   ```

3. **Generate embeddings** - Enable semantic search:
   ```powershell
   curl -X POST "http://127.0.0.1:8000/embeddings/generate?limit=500"
   ```

4. **Ask questions** - Query your knowledge base:
   ```powershell
   curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "What are my main project ideas?"}'
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│  /search    /ask     /file    /embeddings/generate          │
└──────┬────────┬────────┬──────────────┬─────────────────────┘
       │        │        │              │
       ▼        ▼        ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐
│   FTS5   │ │   RAG    │ │ Database │ │  Embedding Service   │
│  Search  │ │ Service  │ │  (db.py) │ │   (embeddings.py)    │
└──────────┘ └────┬─────┘ └──────────┘ └──────────┬───────────┘
                  │                               │
                  ▼                               ▼
           ┌──────────────┐              ┌───────────────────┐
           │Vector Search │              │   LLM Provider    │
           │(vector_search)│              │ OpenAI/Gemini/   │
           └──────┬───────┘              │     Ollama        │
                  │                      └───────────────────┘
                  ▼
           ┌──────────────┐
           │  LLM Chat    │
           │  Generation  │
           └──────────────┘
```

## Phase 2 Roadmap

Future enhancements planned:

- **Auto-embed on index**: Automatically generate embeddings when files are indexed
- **Conversation history**: Multi-turn conversations with context
- **Streaming responses**: Stream LLM output for better UX
- **FAISS integration**: Faster vector search for large vaults
- **Web UI**: Simple web interface for Q&A

## Troubleshooting

### "VAULT_PATH must be set to a valid directory"

Make sure your `.env` file exists and `VAULT_PATH` points to a valid folder:

```powershell
# Check if the path exists
Test-Path "C:\Users\YourName\Documents\ObsidianVault"
```

### Database Locked Errors

Only run one instance of the daemon at a time. If you need to reset:

```powershell
Remove-Item second_brain.db
```

### File Changes Not Detected

The watcher has a 1-second debounce by default. Wait a moment after saving.

## License

MIT License - Feel free to use and modify.
