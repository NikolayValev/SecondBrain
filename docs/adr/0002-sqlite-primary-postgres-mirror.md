# ADR 0002: SQLite Is Primary; PostgreSQL Is Best-Effort Mirror

- Status: Accepted
- Date: 2026-02-25

## Context

SecondBrain runs locally with SQLite as the operational datastore (`app/db.py`) for indexing, search, embeddings, and conversations.
PostgreSQL (`app/db_postgres.py`) exists for remote/frontend consumption and is synchronized through sync services.

Current code paths mirror to PostgreSQL opportunistically (for example conversation writes and sync jobs), and should not block local operation when PostgreSQL is unavailable or partially failing.

## Decision

1. Treat SQLite as the source of truth for runtime behavior.
2. Treat PostgreSQL as an eventually consistent mirror for integration/read use cases.
3. Keep PostgreSQL sync best-effort and non-blocking for local user workflows.
4. Preserve explicit sync APIs and stats so consumers can observe and repair drift.

## Consequences

- Local reliability remains high even when PostgreSQL is offline.
- SQLite and PostgreSQL records may be temporarily out of sync.
- Conversation identity/ordering assumptions across stores must be handled carefully.
- Agents changing sync behavior must prioritize non-blocking local writes and clear error visibility.
