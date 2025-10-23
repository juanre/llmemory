## Architecture review: llmemory, llmbridge, pgdbm

### Scope
- Analyze architecture, DB/migrations strategy, and integration across `llmemory`, `llmbridge`, and `pgdbm`.
- Identify issues affecting safe multi-service usage with shared pools and schema isolation.
- Provide prioritized path forward and implement initial fixes.

### Current architecture
- llmemory
  - Library facade `LLMemory` supports standalone or injected `pgdbm.AsyncDatabaseManager` (shared pool).
  - DB layer `MemoryDatabase` wraps `pgdbm`; runs own migrations and prepared statements; schema-aware via `{{tables.*}}`/`{{schema}}`.
  - Schema: pgvector + HNSW, full-text, hybrid search (RRF), semantic/hierarchical chunking, batch embedding processors.

- llmbridge
  - Service/library for LLM models, pricing/capabilities, and usage tracking.
  - API `LLMBridgeAPI` (discovery, cost calc, validation, health/metrics).
  - DB `LLMDatabase` supports standalone or injected manager; runs own migrations.
  - Providers: Anthropic/Google SDKs with normalization and pricing scrapers.

- pgdbm
  - `AsyncDatabaseManager` on asyncpg with optional shared pool and schema override; query templating (`{{schema}}`, `{{tables.*}}`).
  - `AsyncMigrationManager` with checksums, per-module tracking, dry-run/history; template expansion at execution.

### Findings
- Schema isolation & module-based migrations are clean in `llmemory` and supported by `pgdbm`.
- `llmbridge` migrations hardcode schema and search_path, not using templates; breaks reuse under arbitrary schema.
- `llmbridge.update_model_costs` updates wrong columns; `health_check` references `self.config` even when using injected manager.
- `llmemory` chunk de-dup is too strict: `UNIQUE(content_hash)` blocks identical chunks across different documents/owners.
- Migration tracking location docs inconsistent with code: code stores `schema_migrations` in the manager’s schema if set.
- Concurrency: migrations lack advisory locking, risking cold-start races.
- Minor: pgvector enabling duplicated in migration and runtime (benign).

### Path forward (prioritized)
1) Make `llmbridge` migrations schema-aware
   - Replace hardcoded `llmbridge`/search_path with templates: `{{schema}}` for functions/types and `{{tables.*}}` for tables.
   - Keep `module_name="llmbridge"` for migration isolation.

2) Fix `llmbridge` bugs
   - Use `dollars_per_million_tokens_input/output` in `update_model_costs`.
   - In `health_check`, report schema from `self.db.schema` (works in standalone/injected modes).

3) Relax `llmemory` chunk deduplication
   - New migration to drop the global unique and add `UNIQUE(document_id, content_hash)`.

4) Harden migrations for concurrency
   - Add advisory lock per `module_name` around apply-pending sequence in `AsyncMigrationManager`.

5) Align documentation
   - Clarify that `schema_migrations` is created in the configured schema when one is set (not always `public`).

6) Optional follow-ups
   - Decide single place to enable pgvector (migration vs runtime) and document required privileges.
   - Expand monitoring around pool usage and slow queries using `MonitoredAsyncDatabaseManager` across services.

### Implemented in this change set
- `llmbridge` migrations templated for schema isolation; functions qualified under `{{schema}}`.
- `llmbridge` DB fixes for cost update columns and schema reporting in health.
- New `llmemory` migration to relax chunk dedup constraint to `(document_id, content_hash)`.
- Advisory locking added to `pgdbm` migration application to prevent concurrent runners.
- Docs updated to reflect migration tracking table location.

### Rollout notes
- Applying the new `llmemory` migration alters a unique constraint; ensure staging data does not rely on the previous global constraint.
- `llmbridge` migrations will now run in the assigned schema under shared-pool deployments; if existing installs used the hardcoded `llmbridge` schema, keep the same schema name in production to avoid moving data.
- Advisory locks are connection-safe; they serialize migration application only and do not affect normal runtime.


