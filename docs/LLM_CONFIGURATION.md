# LLM Configuration Guide

This guide reflects current provider logic in `app/config.py`, `app/llm.py`, and `app/embeddings.py`.

## Supported Chat Providers

- `openai`
- `gemini`
- `ollama`
- `anthropic`

Set via:

```env
LLM_PROVIDER=openai
```

## Embedding Provider Selection

By default, embeddings use the same provider as `LLM_PROVIDER`.

Optional override:

```env
EMBEDDING_PROVIDER=openai
```

Valid override values: `openai`, `gemini`, `ollama`.

This is most useful with `LLM_PROVIDER=anthropic` because Anthropic does not provide an embedding API in this codebase.

## Common Settings

```env
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

## Provider-Specific Settings

## OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=https://api.openai.com/v1
```

## Gemini

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-pro
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

## Ollama

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Anthropic

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
# Strongly recommended:
EMBEDDING_PROVIDER=openai
```

If `EMBEDDING_PROVIDER` is not set with Anthropic, the code tries auto-detection in this order:

1. `openai`
2. `gemini`
3. `ollama`

If none initialize successfully, embedding calls fail.

## Runtime Behavior

- Global default provider instance comes from `get_llm_provider()`.
- Per-request provider selection in RAG routes uses `get_provider_by_name(...)`.
- `/health` checks provider availability and reports errors.
- `/config` returns available providers, model catalogs, and defaults.

## Validation Rules

`Config.validate_llm_config()` enforces:

- `LLM_PROVIDER` must be one of `openai|gemini|ollama|anthropic`.
- Required API key must exist for `openai`, `gemini`, and `anthropic`.
- `ollama` has no key requirement, but runtime availability still depends on reachable server.

## Quick Checks

1. Start API and call:
   - `GET /health`
   - `GET /config`
2. Verify provider and model defaults.
3. Generate embeddings:
   - `POST /embeddings/generate`
4. Ask a question:
   - `POST /ask`

## Common Misconfigurations

- `LLM_PROVIDER=anthropic` without a usable embedding fallback.
- Missing provider API keys.
- Ollama running on a different URL than `OLLAMA_BASE_URL`.
- Assuming `/ask` model override also changes embedding provider (it does not).
