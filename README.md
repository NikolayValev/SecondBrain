# Second Brain

A local daemon that indexes an Obsidian vault and exposes a REST API for search and file access. Designed for Windows, with future LLM/RAG integration in mind.

## Features

- **File Watching**: Monitors your vault for changes in real-time
- **Full-Text Search**: FTS5-powered keyword search with BM25 ranking
- **Markdown Parsing**: Extracts titles, headings, tags, and links
- **REST API**: FastAPI-based endpoints for integration
- **Incremental Indexing**: Only re-indexes changed files

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

## API Documentation

FastAPI provides interactive documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Project Structure

```
SecondBrain/
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI application
│   ├── config.py     # Configuration management
│   ├── db.py         # SQLite database with FTS5
│   ├── parser.py     # Markdown parsing
│   ├── indexer.py    # File indexing logic
│   └── watcher.py    # File system watcher
├── .env              # Environment configuration
├── .env.example      # Example configuration
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Database Schema

The SQLite database includes:

- **files**: File metadata (path, mtime, title, content)
- **sections**: Headings and their content
- **tags**: Unique tag names
- **file_tags**: File-tag associations
- **links**: Outbound links from files
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

## Phase 2 Roadmap

This codebase is structured to support future enhancements:

- **Embedding Generation**: Add a module for generating text embeddings
- **Vector Database**: Store embeddings in a vector database (e.g., ChromaDB, FAISS)
- **RAG Endpoint**: Implement `/ask` endpoint for LLM-powered Q&A

### Extension Points

- `app/embeddings.py` - For embedding model integration
- `app/vector_db.py` - For vector storage
- `app/rag.py` - For RAG pipeline

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
