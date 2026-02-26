# Architecture Decision Records (ADR)

This folder stores concise, durable architectural decisions that agents should check before making cross-cutting changes.

## Index

- [0001-layered-api-routes-models-services](./0001-layered-api-routes-models-services.md)
- [0002-sqlite-primary-postgres-mirror](./0002-sqlite-primary-postgres-mirror.md)
- [0003-public-api-security-baseline](./0003-public-api-security-baseline.md)
- [0004-request-size-guardrails-public-api](./0004-request-size-guardrails-public-api.md)
- [0005-per-request-llm-tuning-parameters](./0005-per-request-llm-tuning-parameters.md)

## When To Add A New ADR

Add an ADR when a decision changes behavior across multiple modules, especially around:

- API boundaries and layering
- Data ownership and sync semantics
- Authentication and security model
- RAG retrieval/generation policy
- Compatibility constraints that future refactors might break
