# Migrations

llmemory uses pgdbm migrations and applies them automatically during `LLMemory.initialize()`.

## Behavior

- Migrations run into the configured schema (default: `llmemory`).
- Running migrations multiple times is safe (idempotent).
- llmemory migrations use pgdbm checksums; do not edit released migration files in place.

## Running Migrations as a Deployment Step

If you prefer applying migrations before starting your app, run them explicitly:

```python
from llmemory.db import MemoryDatabase, create_memory_db_manager

db_manager = await create_memory_db_manager("postgresql://...", schema="llmemory")
db = MemoryDatabase(db_manager)
await db.apply_migrations()
await db.close()
```

See `docs/integration-guide.md` for background and configuration details.

