# Installation Guide

## Requirements

Before installing llmemory, ensure you have:

- Python 3.10 or higher
- PostgreSQL 14+ with pgvector extension
- OpenAI API key (if using OpenAI embeddings) or local model support

## PostgreSQL Setup

### 1. Install PostgreSQL with pgvector

#### macOS (using Homebrew)
```bash
# Install PostgreSQL
brew install postgresql@14

# Install pgvector
brew install pgvector

# Start PostgreSQL
brew services start postgresql@14
```

#### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-14 postgresql-server-dev-14

# Install pgvector
cd /tmp
git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Docker
```bash
# Use the official pgvector image
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg14
```

### 2. Enable pgvector Extension

Connect to your database and enable the extension:

```sql
-- Connect to your database
psql -U postgres -d your_database

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## Python Package Installation

### Using uv (recommended)

```bash
# Basic installation
uv add llmemory

# With monitoring support
uv add "llmemory[monitoring]"

# With caching support
uv add "llmemory[cache]"

# With local embedding support
uv add "llmemory[local]"

# All optional dependencies
uv add "llmemory[monitoring,cache,local]"
```

### Using pip

```bash
# Basic installation
pip install llmemory

# With monitoring support
pip install "llmemory[monitoring]"

# With caching support
pip install "llmemory[cache]"

# With local embedding support
pip install "llmemory[local]"

# All optional dependencies
pip install "llmemory[monitoring,cache,local]"
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/llmemory.git
cd llmemory

# Install with uv (recommended)
uv add -e .

# Or with pip
pip install -e .
```

## Configuration

### 1. Environment Variables

Create a `.env` file or set environment variables:

```bash
# Database connection
DATABASE_URL=postgresql://user:password@localhost:5432/mydb

# OpenAI configuration (if using OpenAI embeddings)
AWORD_OPENAI_API_KEY=sk-...
AWORD_OPENAI_MODEL=text-embedding-3-small

# Alternative: Use local embeddings
AWORD_EMBEDDING_PROVIDER=local-minilm
AWORD_LOCAL_MODEL=all-MiniLM-L6-v2
AWORD_LOCAL_DEVICE=cpu  # or cuda if you have GPU

# Optional: Database pool settings
AWORD_DB_MAX_POOL_SIZE=20

# Optional: Logging
AWORD_LOG_LEVEL=INFO
```

### 2. Verify Installation

Create a test script to verify your installation:

```python
import asyncio
from llmemory import AwordMemory, DocumentType

async def test_installation():
    # Initialize memory service
    memory = AwordMemory(
        connection_string="postgresql://localhost/testdb",
        openai_api_key="sk-..."  # Or use local embeddings
    )

    try:
        # Initialize database
        await memory.initialize()
        print("✅ Database connection successful")

        # Test document addition
        result = await memory.add_document(
            owner_id="test-owner",
            id_at_origin="test-user",
            document_name="test.txt",
            document_type=DocumentType.GENERAL,
            content="This is a test document."
        )
        print(f"✅ Document added: {result.document_id}")

        # Test search
        results = await memory.search(
            owner_id="test-owner",
            query_text="test"
        )
        print(f"✅ Search successful: {len(results)} results")

        # Cleanup
        await memory.delete_document(
            owner_id="test-owner",
            document_id=result.document_id
        )
        print("✅ Cleanup successful")

    finally:
        await memory.close()
        print("✅ Connection closed")

# Run the test
asyncio.run(test_installation())
```

## Troubleshooting

### Common Issues

#### 1. pgvector Extension Not Found

```
Error: extension "vector" is not available
```

**Solution**: Ensure pgvector is properly installed and the PostgreSQL user has CREATE EXTENSION privileges.

```sql
-- As superuser
GRANT CREATE ON DATABASE your_database TO your_user;
```

#### 2. OpenAI API Key Not Found

```
Error: OpenAI API key not found
```

**Solution**: Set the environment variable or pass it directly:

```python
# Via environment
export AWORD_OPENAI_API_KEY=sk-...

# Or in code
memory = AwordMemory(
    connection_string="...",
    openai_api_key="sk-..."
)
```

#### 3. Connection Pool Exhausted

```
Error: connection pool exhausted
```

**Solution**: Increase pool size or ensure proper connection cleanup:

```python
# Increase pool size
export AWORD_DB_MAX_POOL_SIZE=50

# Always close connections
try:
    await memory.initialize()
    # ... do work ...
finally:
    await memory.close()
```

#### 4. Local Model Download Issues

```
Error: Could not download model
```

**Solution**: Ensure you have internet access or pre-download models:

```python
# Set cache directory
export AWORD_LOCAL_CACHE_DIR=/path/to/models

# Pre-download models
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/path/to/models/all-MiniLM-L6-v2')
```

### Performance Optimization

#### 1. Enable Connection Pooling

```python
# Share pools across services via pgdbm
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import AwordMemory

config = DatabaseConfig(connection_string="postgresql://localhost/db")
pool = await AsyncDatabaseManager.create_shared_pool(config)

service1_db = AsyncDatabaseManager(pool=pool, schema="service1")
service2_db = AsyncDatabaseManager(pool=pool, schema="service2")

service1 = AwordMemory.from_db_manager(service1_db)
service2 = AwordMemory.from_db_manager(service2_db)
```

#### 2. Use Appropriate Indexes

The library automatically creates optimized indexes, but ensure they're used:

```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM document_chunks
WHERE owner_id = 'test'
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
```

#### 3. Enable Monitoring

Install with monitoring support to track performance:

```bash
# Using uv (recommended)
uv add "llmemory[monitoring]"

# Or using pip
pip install "llmemory[monitoring]"
```

Then access metrics at `/metrics` endpoint if using with a web framework.

## Next Steps

- Read the [Quickstart Guide](quickstart.md) for a complete example
- Review the [API Reference](api-reference.md) for detailed documentation
- Check out [Examples](../examples/) for real-world usage patterns
