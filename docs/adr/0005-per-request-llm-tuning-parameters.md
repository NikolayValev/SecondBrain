# ADR 0005: Per-Request LLM Tuning Parameters

- Status: Accepted
- Date: 2026-02-25

## Context

`POST /ask` allowed provider/model selection but did not expose per-request generation tuning controls. This blocked runtime tuning for response quality, determinism, and cost/latency tradeoffs when clients switched models/providers.

## Decision

1. Extend `AskRequest` with optional tuning fields:
   - `temperature`
   - `max_tokens`
   - `top_p`
   - `top_k`
2. Thread those fields through both synchronous and streaming `/ask` paths in `RAGAPIService`.
3. Extend provider chat interfaces (`chat`, `chat_with_usage`, `stream_chat`) to accept model override plus tuning fields.
4. Forward only supported parameters per provider:
   - OpenAI: `temperature`, `max_tokens`, `top_p`
   - Gemini/Ollama/Anthropic: `temperature`, `max_tokens`, `top_p`, `top_k`
   - Unsupported fields are ignored rather than causing request failure.

## Consequences

- Clients can tune generation behavior per request without changing server-wide env defaults.
- Sync and streaming answer paths stay behaviorally aligned for tuning.
- Provider interface surface grows, increasing need for test coverage across providers.
