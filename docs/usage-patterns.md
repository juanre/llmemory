# Usage Patterns

llmemory supports two integration modes:

## Standalone Mode

Let llmemory create and manage its own database connection pool.

```python
from llmemory import LLMemory

memory = LLMemory(connection_string="postgresql://localhost/mydb")
await memory.initialize()
```

## Shared Pool Mode

Reuse a parent application's pool, but isolate llmemory into its own schema.

```python
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import LLMemory

config = DatabaseConfig(connection_string="postgresql://localhost/myapp")
shared_pool = await AsyncDatabaseManager.create_shared_pool(config)

llmemory_db = AsyncDatabaseManager(pool=shared_pool, schema="llmemory")
memory = LLMemory.from_db_manager(llmemory_db)
await memory.initialize()
```

See `docs/integration-guide.md` for details and production patterns.

