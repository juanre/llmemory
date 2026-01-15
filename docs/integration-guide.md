# Integration Guide

Complete guide for integrating llmemory into your application.

## Usage Patterns

llmemory supports two integration modes:

### 1. Standalone Mode

llmemory creates and manages its own database connection:

```python
from llmemory import LLMemory, DocumentType

memory = LLMemory(
    connection_string="postgresql://localhost/mydb",
    openai_api_key="sk-..."
)
await memory.initialize()  # Creates pool, runs migrations

result = await memory.add_document(
    owner_id="workspace-1",
    id_at_origin="user-123",
    document_name="report.pdf",
    document_type=DocumentType.PDF,
    content="..."
)

await memory.close()
```

**Behavior:**
- Creates own connection pool
- Runs migrations in configured schema (default: `llmemory`)
- Tables: `{schema}.documents`, `{schema}.document_chunks`, etc.

### 2. Shared Pool Mode (Sublibrary)

Parent application provides AsyncDatabaseManager with schema already set:

```python
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import LLMemory

# Parent creates shared pool
config = DatabaseConfig(connection_string="postgresql://localhost/myapp")
shared_pool = await AsyncDatabaseManager.create_shared_pool(config)

# Create schema-isolated manager for llmemory
llmemory_db = AsyncDatabaseManager(pool=shared_pool, schema="llmemory")

# Pass to llmemory
memory = LLMemory.from_db_manager(llmemory_db)
await memory.initialize()  # Uses parent's pool, runs migrations in "llmemory" schema
```

**Behavior:**
- Uses parent's connection pool
- Runs migrations into parent-specified schema
- Tables: `{parent_schema}.documents`, `{parent_schema}.document_chunks`, etc.
- Does not close parent's pool on `close()`

## Migrations

llmemory uses pgdbm's migration system with automatic execution.

**Migration Behavior:**
- Runs automatically during `initialize()`
- Uses `{{tables.tablename}}` template syntax for schema awareness
- Tracked in `schema_migrations` table with module name `llmemory`
- Idempotent and safe to run multiple times

**Manual Migration Control:**

Migrations are applied automatically during `initialize()` and are safe to run multiple times.

If you want to run migrations as a separate deployment step, apply them explicitly:

```python
from llmemory.db import MemoryDatabase, create_memory_db_manager

db_manager = await create_memory_db_manager("postgresql://...", schema="llmemory")
db = MemoryDatabase(db_manager)
await db.apply_migrations()
await db.close()
```

**Migration File Location:**
`src/llmemory/migrations/001_complete_schema.sql`

## Framework Integration

### Advanced Retrieval Configuration

llmemory ships with three opt-in levers you can enable globally or per request:

- **Query expansion** – generates semantic + keyword variants before retrieval.
- **Reranking** – refines the final hit list using a pluggable reranker callback.
- **Chunk summaries** – captures short synopses during ingestion and returns them with each search result.

Enable them globally via configuration:

```python
from llmemory import LLMemory, LLMemoryConfig

config = LLMemoryConfig()
config.chunking.enable_chunk_summaries = True
config.search.enable_query_expansion = True
config.search.enable_rerank = True

memory = LLMemory(connection_string="postgresql://localhost/mydb", config=config)
await memory.initialize()
```

Or toggle them per-call when you need extra precision:

```python
results = await memory.search(
    owner_id=owner_id,
    query_text=query_text,
    search_type="hybrid",
    query_expansion=True,
    rerank=True,
)

for hit in results:
    print(hit.summary, hit.score)
```

To integrate a learned reranker (e.g., Cohere ReRank, OpenAI ReRank, Anthropic Claude), pass a scoring callback when creating `LLMemory`. Install the local cross-encoder dependencies if you plan to run the built-in reranker:

```bash
uv add "llmemory[reranker-local]"
```

The callback receives the query text and the candidate `SearchResult` objects, and should return an ordered list of scores:

```python
from llmemory.models import SearchResult
from llmemory.reranker import CrossEncoderReranker, RerankerService

async def cohere_rerank(query: str, results: list[SearchResult]) -> list[float]:
    payload = {
        "query": query,
        "documents": [r.content for r in results],
    }
    response = await cohere_client.rerank(**payload)
    return [doc["relevance_score"] for doc in response["results"]]

memory = LLMemory(
    connection_string="postgresql://localhost/mydb",
    config=config,
)
memory._reranker = RerankerService(memory.config.search, score_callback=cohere_rerank)
```

You can also tune pgvector's HNSW index for different latency/recall trade-offs by setting a preset before initialising llmemory:

```bash
# Faster lookups with slightly lower recall
LLMEMORY_HNSW_PROFILE=fast

# Higher recall at increased query cost
LLMEMORY_HNSW_PROFILE=accurate
```

For OpenAI-based reranking (using the same API key configured for embeddings), set:

```bash
LLMEMORY_RERANK_PROVIDER=openai
LLMEMORY_RERANK_MODEL=gpt-4.1-mini
```

Every search is logged to `search_history` with diagnostic metadata (`query_variants`, `backend`, latency breakdowns, rerank flags), so you can monitor adoption and tune thresholds in production.

For offline evaluation, install the benchmarking extras (`uv add "llmemory[bench]"`) and run:

```bash
python bench/beir_runner.py nq --dataset-dir ./datasets --connection postgresql://localhost/llmemory_bench --query-expansion --rerank
```

The harness ingests the specified BEIR dataset into llmemory, executes the configured pipeline, and prints standard metrics (nDCG, Recall, MRR).

### FastAPI

```python
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from llmemory import LLMemory, DocumentType

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.memory = LLMemory(
        connection_string="postgresql://localhost/mydb",
        openai_api_key="sk-..."
    )
    await app.state.memory.initialize()
    yield
    await app.state.memory.close()

app = FastAPI(lifespan=lifespan)

async def get_memory(request: Request):
    return request.app.state.memory

@app.post("/documents")
async def add_document(
    owner_id: str,
    content: str,
    memory: LLMemory = Depends(get_memory)
):
    result = await memory.add_document(
        owner_id=owner_id,
        id_at_origin="api",
        document_name="uploaded.txt",
        document_type=DocumentType.TEXT,
        content=content
    )
    return {"document_id": str(result.document.document_id)}

@app.get("/search")
async def search(
    owner_id: str,
    query: str,
    memory: LLMemory = Depends(get_memory)
):
    results = await memory.search(
        owner_id=owner_id,
        query_text=query,
        search_type="hybrid"
    )
    return {"results": [{"content": r.content, "score": r.score} for r in results]}
```

### FastAPI with Shared Pool

When multiple services share a database:

```python
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import LLMemory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create shared pool once
    config = DatabaseConfig(
        connection_string="postgresql://localhost/myapp",
        min_connections=20,
        max_connections=100
    )
    shared_pool = await AsyncDatabaseManager.create_shared_pool(config)

    # Create schema-isolated managers
    app.state.dbs = {
        'users': AsyncDatabaseManager(pool=shared_pool, schema="users"),
        'llmemory': AsyncDatabaseManager(pool=shared_pool, schema="llmemory"),
    }

    # Initialize llmemory with shared manager
    app.state.memory = LLMemory.from_db_manager(app.state.dbs['llmemory'])
    await app.state.memory.initialize()

    yield

    await app.state.memory.close()
    await shared_pool.close()

app = FastAPI(lifespan=lifespan)
```

### Django

```python
from django.apps import AppConfig
from asgiref.sync import async_to_sync
from llmemory import LLMemory

class MyAppConfig(AppConfig):
    def ready(self):
        # Initialize on Django startup
        memory = LLMemory(connection_string="postgresql://localhost/mydb")
        async_to_sync(memory.initialize)()
        self.memory = memory
```

### Flask

```python
from flask import Flask, g
from llmemory import LLMemory

app = Flask(__name__)
memory = None

@app.before_first_request
async def init_memory():
    global memory
    memory = LLMemory(connection_string="postgresql://localhost/mydb")
    await memory.initialize()

@app.before_request
def set_memory():
    g.memory = memory

@app.route('/search')
async def search():
    results = await g.memory.search(
        owner_id=request.args['owner_id'],
        query_text=request.args['q']
    )
    return jsonify([r.to_dict() for r in results])
```

## Multi-Tenant Isolation

llmemory provides tenant isolation via `owner_id`:

```python
# Each tenant gets isolated data via owner_id
result = await memory.add_document(
    owner_id="tenant-123",  # Tenant identifier
    id_at_origin="user-456",
    document_name="doc.pdf",
    document_type=DocumentType.PDF,
    content="..."
)

# Searches are automatically scoped to owner
results = await memory.search(
    owner_id="tenant-123",  # Only searches this tenant's data
    query_text="project timeline"
)
```

**Note:** `owner_id` provides logical isolation within a shared schema. For stronger isolation, use separate schemas (shared pool pattern) with one schema per tenant.

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://localhost/mydb

# OpenAI (if using OpenAI embeddings)
OPENAI_API_KEY=sk-...

# Optional: Override defaults
LLMEMORY_SCHEMA=custom_schema
LLMEMORY_MAX_CHUNK_SIZE=2000
```

### Programmatic Configuration

```python
from llmemory import LLMemory, LLMemoryConfig

config = LLMemoryConfig()
config.chunking.chunk_size = 1500
config.chunking.chunk_overlap = 300

memory = LLMemory(
    connection_string="postgresql://localhost/mydb",
    config=config
)
```

## Production Considerations

### Connection Pooling

For production with multiple services, use shared pool pattern to avoid connection exhaustion:

```python
# ONE shared pool for entire application
shared_pool = await AsyncDatabaseManager.create_shared_pool(
    DatabaseConfig(
        connection_string="postgresql://production/db",
        min_connections=50,
        max_connections=200
    )
)

# Each service gets schema-isolated manager
llmemory_db = AsyncDatabaseManager(pool=shared_pool, schema="llmemory")
users_db = AsyncDatabaseManager(pool=shared_pool, schema="users")

# Initialize llmemory
memory = LLMemory.from_db_manager(llmemory_db)
await memory.initialize()
```

### Error Handling

```python
from llmemory import ValidationError, DatabaseError, EmbeddingError

try:
    result = await memory.add_document(...)
except ValidationError as e:
    # Handle validation errors (bad input)
    return {"error": f"Validation failed: {e.field} - {e.message}"}
except DatabaseError as e:
    # Handle database errors
    logger.error(f"Database error: {e}")
    return {"error": "Database operation failed"}
except EmbeddingError as e:
    # Handle embedding generation errors
    logger.warning(f"Embedding failed: {e}")
    # Document still created, embeddings can be generated later
```

## Testing Integration

Use llmemory's test fixtures:

```python
# conftest.py
from pgdbm.fixtures.conftest import *
from llmemory.testing import *

# Test with fixture
async def test_document_workflow(memory_library):
    result = await memory_library.add_document(
        owner_id="test",
        id_at_origin="origin",
        document_name="test.txt",
        document_type="text",
        content="test content"
    )
    assert result.document.document_id is not None
```

See [Testing Guide](testing-guide.md) for complete testing patterns.
