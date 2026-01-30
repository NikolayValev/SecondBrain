# RAG (Retrieval-Augmented Generation) Guide

This document explains how the RAG system works in Second Brain and how to use it effectively.

## Overview

RAG combines document retrieval with LLM generation to answer questions based on your knowledge base. Instead of relying solely on the LLM's training data, RAG retrieves relevant content from your notes and uses it as context.

## How It Works

```
User Question
     │
     ▼
┌────────────────┐
│ Embed Question │  ← Convert question to vector
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Vector Search  │  ← Find similar chunks in your notes
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Build Context  │  ← Combine top chunks into context
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ LLM Generation │  ← Generate answer using context
└───────┬────────┘
        │
        ▼
   Answer + Sources
```

## Pipeline Steps

### 1. Text Chunking

Documents are split into smaller chunks for embedding:

- **Max chunk size**: ~512 tokens (configurable)
- **Overlap**: 50 tokens between chunks for context continuity
- **Splitting strategy**: Paragraph boundaries → sentence boundaries → word boundaries

### 2. Embedding Generation

Each chunk is converted to a vector embedding:

- **OpenAI**: `text-embedding-3-small` (1536 dimensions)
- **Gemini**: `text-embedding-004` (768 dimensions)
- **Ollama**: `nomic-embed-text` (768 dimensions)

### 3. Vector Search

When you ask a question:

1. The question is embedded using the same model
2. Cosine similarity is calculated against all stored embeddings
3. Top-K most similar chunks are retrieved
4. Results are deduplicated by file (max 2 chunks per file)

### 4. Context Building

Retrieved chunks are formatted into a context string:

```
[Source 1: Note Title > Section Heading]
Content from the first chunk...

[Source 2: Another Note > Introduction]
Content from the second chunk...
```

### 5. LLM Generation

The context and question are sent to the LLM:

```
System: You are a helpful assistant that answers questions 
based on the user's personal knowledge base...

User: Context from knowledge base:
---
[Retrieved chunks here]
---

Question: What is...?

Answer based on the context above:
```

## API Usage

### Basic Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are my notes about machine learning?"}'
```

**Response:**

```json
{
  "answer": "Based on your notes, you have several entries about machine learning...",
  "sources": [
    {
      "file_path": "notes/ml-basics.md",
      "file_title": "Machine Learning Basics",
      "section": "Introduction",
      "similarity": 0.847
    },
    {
      "file_path": "notes/projects/image-classifier.md",
      "file_title": "Image Classifier Project",
      "section": "Implementation",
      "similarity": 0.792
    }
  ],
  "query": "What are my notes about machine learning?"
}
```

### Without Sources

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize my project ideas", "include_sources": false}'
```

## Configuration

### RAG Parameters

These are currently configured in `app/rag.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Number of chunks to retrieve |
| `max_context_tokens` | 3000 | Maximum context size |
| `similarity_threshold` | 0.3 | Minimum similarity score |

### Chunker Parameters

Configured in `app/chunker.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_tokens` | 512 | Maximum tokens per chunk |
| `overlap_tokens` | 50 | Overlap between chunks |
| `min_chunk_tokens` | 50 | Minimum viable chunk size |

## Best Practices

### Generating Embeddings

After indexing your vault, generate embeddings:

```bash
# Generate embeddings for all pending chunks
curl -X POST "http://127.0.0.1:8000/embeddings/generate?limit=1000"
```

Check progress:

```bash
curl "http://127.0.0.1:8000/embeddings/stats"
```

Response:

```json
{
  "chunk_count": 1500,
  "embedding_count": 1200,
  "files_with_embeddings": 45,
  "pending_chunks": 300
}
```

### Asking Good Questions

For best results:

1. **Be specific**: "What did I write about React hooks?" vs "Tell me about React"
2. **Use keywords from your notes**: The search uses semantic similarity
3. **Ask about topics you've written about**: RAG can only find what exists

### Handling Large Vaults

For vaults with many notes:

1. Generate embeddings in batches: `?limit=100` then repeat
2. Consider using Ollama locally to avoid API costs
3. The first embedding generation takes longer; subsequent updates are incremental

## Troubleshooting

### "LLM not configured" Error

Make sure you've set your API key:

```ini
# .env file
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

### No Relevant Results

1. Check that embeddings exist: `GET /embeddings/stats`
2. If `embedding_count` is 0, run `POST /embeddings/generate`
3. Try more specific or differently-worded questions

### Slow Response Times

- **OpenAI/Gemini**: Network latency + API processing
- **Ollama**: Depends on your hardware; GPU recommended
- **Vector search**: Currently brute-force; consider FAISS for large vaults

### Outdated Answers

If notes have changed:

1. Reindex: `POST /reindex?full=true`
2. Regenerate embeddings: `POST /embeddings/generate`

## Cost Considerations

### API Usage

| Provider | Embedding Cost | Chat Cost |
|----------|---------------|-----------|
| OpenAI | ~$0.02/1M tokens | ~$5-15/1M tokens |
| Gemini | Free tier available | Free tier available |
| Ollama | Free (local) | Free (local) |

### Recommendations

- **Development/Testing**: Use Ollama (free, private)
- **Production with budget**: Use Gemini (generous free tier)
- **Best quality**: Use OpenAI GPT-4o

## Security Notes

1. **API keys**: Never commit `.env` to version control
2. **Local storage**: Embeddings are stored in `second_brain.db`
3. **Ollama**: All processing stays on your machine
4. **Cloud providers**: Your note content is sent to their APIs
