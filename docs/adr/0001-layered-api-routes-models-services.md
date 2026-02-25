# ADR 0001: Layered API With Routes, Models, and Services

- Status: Accepted
- Date: 2026-02-25

## Context

The application has evolved from route-heavy endpoint handlers toward a layered FastAPI structure:

- `app/main.py` composes the app and lifecycle.
- `app/api/routes/*` defines HTTP handlers.
- `app/api/models/*` defines request/response contracts.
- `app/services/*` holds domain logic.

Without a recorded decision, future edits can regress into mixed concerns (business logic inside routes, contract drift across layers).

## Decision

Adopt and preserve the layered boundary:

1. Keep route handlers thin (validation, dependency wiring, HTTP error mapping).
2. Keep domain behavior in services.
3. Keep API contracts in model modules and evolve them alongside service behavior.
4. Treat changes to endpoint behavior as coordinated updates across route + model + service.

## Consequences

- Improves maintainability and testability by isolating HTTP concerns from business logic.
- Makes refactors safer because layer responsibilities are explicit.
- Requires discipline: endpoint changes now involve multiple files by design.
- Agents should reject shortcuts that place non-trivial logic directly in route modules.
