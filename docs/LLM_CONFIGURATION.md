# LLM Configuration Guide

This document describes the LLM (Large Language Model) configuration system implemented in Second Brain.

## Overview

Second Brain supports three LLM providers:
- **OpenAI** - Cloud-based models (GPT-4o, GPT-4, etc.)
- **Google Gemini** - Cloud-based models (Gemini 1.5 Pro, etc.)
- **Ollama** - Local models (Llama3, Mistral, etc.)

## Configuration

All LLM settings are configured via environment variables in your `.env` file.

### Provider Selection

Set the `LLM_PROVIDER` variable to choose your provider:

```env
LLM_PROVIDER=openai   # Use OpenAI
LLM_PROVIDER=gemini   # Use Google Gemini
LLM_PROVIDER=ollama   # Use Ollama (local)
```

### Common Settings

These settings apply to all providers:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_TEMPERATURE` | `0.7` | Controls randomness (0.0 = deterministic, 1.0 = creative) |
| `LLM_MAX_TOKENS` | `4096` | Maximum tokens in generated responses |

### OpenAI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model to use for chat/completion |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint (for proxies/Azure) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for generating embeddings |

### Google Gemini Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google AI API key |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Model to use for chat/completion |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Model for generating embeddings |

### Ollama Configuration

Ollama runs locally and doesn't require an API key.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Model to use for chat/completion |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model for generating embeddings |

## Usage in Code

### Getting LLM Configuration

```python
from app.config import config

# Get configuration dictionary for the active provider
llm_config = config.get_llm_config()

# Example output for OpenAI:
# {
#     "provider": "openai",
#     "temperature": 0.7,
#     "max_tokens": 4096,
#     "api_key": "sk-...",
#     "model": "gpt-4o",
#     "base_url": "https://api.openai.com/v1",
#     "embedding_model": "text-embedding-3-small"
# }
```

### Validating Configuration

```python
from app.config import config

# Validate LLM configuration before use
try:
    config.validate_llm_config()
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Accessing Individual Settings

```python
from app.config import Config

# Direct access to settings
provider = Config.LLM_PROVIDER
temperature = Config.LLM_TEMPERATURE

# Provider-specific settings
if Config.LLM_PROVIDER == "openai":
    api_key = Config.OPENAI_API_KEY
    model = Config.OPENAI_MODEL
```

## Example Configurations

### OpenAI Setup

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
LLM_TEMPERATURE=0.7
```

### Gemini Setup

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-1.5-pro
LLM_TEMPERATURE=0.7
```

### Ollama Setup (Local)

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Implementation Details

### Files Modified

- **`app/config.py`** - Added LLM configuration class attributes and methods
- **`.env.example`** - Added documentation for all LLM environment variables

### Config Class Additions

| Attribute/Method | Type | Description |
|------------------|------|-------------|
| `LLM_PROVIDER` | `str` | Active provider name |
| `OPENAI_API_KEY` | `str` | OpenAI API key |
| `OPENAI_MODEL` | `str` | OpenAI model name |
| `OPENAI_BASE_URL` | `str` | OpenAI API endpoint |
| `OPENAI_EMBEDDING_MODEL` | `str` | OpenAI embedding model |
| `GEMINI_API_KEY` | `str` | Gemini API key |
| `GEMINI_MODEL` | `str` | Gemini model name |
| `GEMINI_EMBEDDING_MODEL` | `str` | Gemini embedding model |
| `OLLAMA_BASE_URL` | `str` | Ollama server URL |
| `OLLAMA_MODEL` | `str` | Ollama model name |
| `OLLAMA_EMBEDDING_MODEL` | `str` | Ollama embedding model |
| `LLM_TEMPERATURE` | `float` | Generation temperature |
| `LLM_MAX_TOKENS` | `int` | Max output tokens |
| `validate_llm_config()` | method | Validates provider config |
| `get_llm_config()` | method | Returns provider config dict |

## Notes

1. **API Keys**: Never commit API keys to version control. Use `.env` files.
2. **Ollama**: Must have Ollama installed and running locally. See [ollama.ai](https://ollama.ai).
3. **OpenAI Base URL**: Can be modified to use Azure OpenAI or compatible proxies.
4. **Validation**: Always call `validate_llm_config()` at application startup.

## Related Documentation

- [RAG Guide](RAG_GUIDE.md) - How to use the RAG Q&A system
- [API Reference](API_REFERENCE.md) - Complete API documentation
