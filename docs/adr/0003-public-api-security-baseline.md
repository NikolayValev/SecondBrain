# ADR 0003: Public API Security Baseline

- Status: Accepted
- Date: 2026-02-25

## Context

SecondBrain is exposed via a public endpoint. Existing defaults prioritized local development convenience and could leak unnecessary information or increase attack surface:

- `/config` was public by default.
- API key comparison used direct equality.
- Host allow-list and standard security headers were not enforced.
- File reads accepted unsanitized relative paths.
- Several routes returned raw exception details in HTTP responses.
- Expensive endpoints lacked built-in rate limiting.

## Decision

Adopt a secure-by-default baseline for public deployments:

1. Keep only `/health` public by default; expose `/config` and docs only via explicit env flags.
2. Use constant-time API key comparison.
3. Enforce trusted host validation with configurable `ALLOWED_HOSTS`.
4. Add default response security headers.
5. Reject absolute and traversal file paths before file reads.
6. Return generic 500-level error messages to clients while logging details server-side.
7. Provide deployment flags in `.env.example` (`REQUIRE_API_KEY`, docs/config exposure, debug, host/origin allow-lists).
8. Add a startup security self-check report with fail-fast behavior in public mode.
9. Enforce in-memory per-identity rate limiting with tighter limits for expensive endpoints.

## Consequences

- Reduces accidental information disclosure and request forgery surface.
- Keeps production behavior safer without removing local-dev flexibility.
- Requires deliberate configuration for public docs/config access.
- Requires key management discipline for public deployments.
- Provides an explicit `/security/self-check` report for runtime verification and operations checks.
- Adds a local rate-limit guardrail even when upstream edge controls are misconfigured.
