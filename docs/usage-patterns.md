# Usage Patterns Guide

This guide explains the three main patterns for using llmemory and how it handles database connections and migrations in each case.

## Understanding llmemory's Architecture

llmemory is designed as a **self-contained library** that:
- Manages its own database schema and tables
- Runs its own migrations automatically when initialized
- Can work standalone or share a connection pool with other services
- Uses pgdbm for schema isolation and migration management

## The Three Usage Patterns

### Pattern 1: Standalone Usage (Simple Applications)

When llmemory is used directly by a simple application:

```python
from llmemory import LLMemory, DocumentType

async def main():
    # llmemory creates and manages its own connection
    memory = LLMemory(
        connection_string="postgresql://localhost/myapp",
        openai_api_key="sk-..."
    )

    # Initialize - this will:
    # 1. Create the connection pool
    # 2. Run migrations to create/update tables
    # 3. Set up embedding providers
    await memory.initialize()

    # Use llmemory
    result = await memory.add_document(
        owner_id="app-1",
        id_at_origin="user-123",
        document_name="report.pdf",
        document_type=DocumentType.PDF,
        content="..."
    )

    # Clean up
    await memory.close()
```

**Key Points:**
- llmemory owns the database connection
- Migrations run automatically during `initialize()`
- Tables are created in the default schema (usually 'public')
- Simple and straightforward for small applications

### Pattern 2: Used by Another Library/Module

When a library uses llmemory internally (e.g., a "document-processor" library):

```python
# document_processor/processor.py
from typing import Optional
from llmemory import LLMemory, DocumentType
from pgdbm import AsyncDatabaseManager

class DocumentProcessor:
    """A library that uses llmemory internally."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        db_manager: Optional[AsyncDatabaseManager] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize processor with flexible database options.

        Args:
            connection_string: For standalone mode
            db_manager: For shared pool mode (from parent app)
            openai_api_key: OpenAI API key
        """
        self._external_db = db_manager is not None

        if db_manager:
            # Use provided db manager (Pattern 3)
            self.memory = LLMemory.from_db_manager(
                db_manager,
                openai_api_key=openai_api_key
            )
        else:
            # Create own connection (Pattern 1)
            self.memory = LLMemory(
                connection_string=connection_string,
                openai_api_key=openai_api_key
            )

    async def initialize(self):
        """Initialize the processor and llmemory."""
        # llmemory will run its own migrations
        await self.memory.initialize()

    async def process_document(self, content: str, metadata: dict):
        """Process a document using llmemory."""
        # Your processing logic here
        result = await self.memory.add_document(
            owner_id=metadata.get('workspace_id', 'default'),
            id_at_origin=metadata.get('user_id', 'system'),
            document_name=metadata.get('filename', 'untitled'),
            document_type=DocumentType.TEXT,
            content=processed_content
        )
        return result

    async def close(self):
        """Clean up resources."""
        await self.memory.close()
```

**Key Points:**
- The library supports both standalone and shared pool modes
- It passes configuration downstream to llmemory
- llmemory still manages its own migrations
- The library doesn't need to know about llmemory's internal structure

### Pattern 3: Used by a Final Application (Most Common)

When a final application (like agent-engine) uses multiple services with a shared connection pool:

```python
# main.py - Final application
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import LLMemory
from document_processor import DocumentProcessor  # From Pattern 2

async def setup_services():
    """Set up all services with shared connection pool."""

    # 1. Create shared connection pool
    config = DatabaseConfig(
        connection_string="postgresql://localhost/myapp",
        min_connections=50,   # Total for all services
        max_connections=100
    )
    shared_pool = await AsyncDatabaseManager.create_shared_pool(config)

    # 2. Create schema-isolated db managers for each service
    memory_db = AsyncDatabaseManager(pool=shared_pool, schema="llmemory")
    processor_db = AsyncDatabaseManager(pool=shared_pool, schema="doc_processor")
    auth_db = AsyncDatabaseManager(pool=shared_pool, schema="auth")

    # 3. Initialize services with their db managers
    llmemory = LLMemory.from_db_manager(
        memory_db,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # The processor library also gets a db manager
    doc_processor = DocumentProcessor(
        db_manager=processor_db,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 4. Initialize all services (each runs its own migrations)
    await llmemory.initialize()  # Creates tables in llmemory schema
    await doc_processor.initialize()   # Creates its tables + memory tables in doc_processor schema

    return {
        'pool': shared_pool,
        'memory': llmemory,
        'processor': doc_processor,
    }

# FastAPI example
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    services = await setup_services()
    app.state.services = services

    yield

    # Shutdown
    await services['memory'].close()
    await services['processor'].close()
    await services['pool'].close()

app = FastAPI(lifespan=lifespan)
```

**Key Points:**
- The application creates one shared connection pool
- Each service gets a schema-isolated db manager
- llmemory runs migrations in its assigned schema
- Tables are namespaced (e.g., `llmemory.documents`)
- Services don't interfere with each other

## How Migrations Work

### The Magic of pgdbm

llmemory uses pgdbm's migration system with these key features:

1. **Module-based tracking**: Migrations are tracked by module name (`aword_memory`)
2. **Schema isolation**: Tables use `{{tables.tablename}}` syntax for schema awareness
3. **Automatic execution**: Migrations run during `initialize()` (single initial migration file)

### Migration Flow

```python
# Inside llmemory during initialize()
migrations_path = Path(__file__).parent / "migrations"
migration_manager = AsyncMigrationManager(
    db_manager,
    migrations_path=str(migrations_path),
    module_name="aword_memory"  # Unique identifier
)
await migration_manager.apply_pending_migrations()
```

### What Happens in Each Pattern

**Pattern 1 (Standalone):**
- Creates tables in default schema (usually 'public')
- Example: `public.documents`, `public.document_chunks`

**Pattern 2 (Library):**
- Depends on how the library is initialized
- If standalone: same as Pattern 1
- If given db_manager: same as Pattern 3

**Pattern 3 (Shared Pool):**
- Creates tables in assigned schema
- Example: `llmemory.documents`, `llmemory.document_chunks`
- Migrations tracked in the assigned schema's `schema_migrations` with module_name

## Why This Architecture?

### 1. Encapsulation
- llmemory manages its own schema - users don't need to know internal structure
- Migrations are packaged with the code that uses them
- Updates are transparent to users

### 2. Flexibility
- Works standalone for simple cases
- Supports enterprise patterns with shared pools
- Libraries can pass through both modes

### 3. Safety
- Schema isolation prevents table conflicts
- Module-based migration tracking prevents re-runs
- Each service manages only its own tables

### 4. Efficiency
- Shared pools reduce total database connections
- Services share resources but maintain isolation
- Scales well for microservice architectures

## Common Questions

### Q: What if I need custom schema names?

Configure the schema in the LLMemoryConfig:

```python
from llmemory import LLMemoryConfig

config = LLMemoryConfig()
config.database.schema_name = "my_custom_schema"

memory = LLMemory(
    connection_string="...",
    config=config
)
```

### Q: Can I disable automatic migrations?

No, and you shouldn't need to. llmemory needs its tables to function, and migrations are idempotent (safe to run multiple times).

### Q: What about migration conflicts?

pgdbm tracks migrations by module name, so multiple instances of llmemory will coordinate correctly. The first one runs migrations, others skip them.

### Q: How do I handle migration permissions?

The database user needs:
- CREATE SCHEMA (for schema creation)
- CREATE TABLE (for migrations)
- Standard CRUD permissions

For production with restricted permissions, run migrations during deployment with an admin user, then use a restricted user for runtime.

## Best Practices

1. **Choose the right pattern** based on your architecture
2. **Use shared pools** for multi-service applications
3. **Let llmemory manage its schema** - don't modify its tables directly
4. **Monitor connection pool usage** in production
5. **Use consistent schema naming** across your organization

## Example Projects

See the `examples/` directory for complete examples:
- `simple_usage.py` - Pattern 1 (standalone)
- `shared_pool_example.py` - Pattern 3 (shared pool)
- `production_integration.py` - Real-world patterns
