# ADR 0004: Request Size Guardrails For Public API

- Status: Accepted
- Date: 2026-02-25

## Context

Rate limiting reduces request frequency, but it does not cap payload size per request. For public deployments, oversized request bodies can still cause avoidable memory/CPU pressure and degrade availability.

## Decision

1. Add a global middleware check that rejects oversized request bodies with `413 Payload Too Large`.
2. Make the limit configurable through `MAX_REQUEST_BYTES` (default: `1048576`, 1 MiB).
3. Return explicit headers on rejection:
   - `X-Max-Request-Bytes`
   - `X-Request-Bytes`
4. Include `MAX_REQUEST_BYTES` in startup/runtime security self-checks, failing public mode when configured too high.

## Consequences

- Reduces denial-of-service exposure from oversized payloads.
- Makes payload limits explicit and auditable in config and self-check reports.
- Requires operators to intentionally raise the limit when a legitimate use case needs larger bodies.
