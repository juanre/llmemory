# Installation Guide

## Requirements

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key (optional - can use local embeddings)

## Install llmemory

```bash
pip install llmemory

# Or with optional dependencies
pip install "llmemory[monitoring]"  # Prometheus metrics
pip install "llmemory[local]"       # Local embedding models
```

## PostgreSQL with pgvector

### Docker (Easiest)

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg14
```

### macOS

```bash
brew install postgresql@14 pgvector
brew services start postgresql@14
```

### Ubuntu/Debian

```bash
sudo apt install postgresql-14 postgresql-server-dev-14

# Install pgvector
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
```

### Enable pgvector Extension

```sql
CREATE EXTENSION vector;
```

llmemory attempts this automatically during initialization, but may require superuser privileges.

## Verify Installation

```python
import asyncio
from llmemory import LLMemory

async def test():
    memory = LLMemory(connection_string="postgresql://localhost/postgres")
    await memory.initialize()
    print("llmemory installed successfully")
    await memory.close()

asyncio.run(test())
```

## Configuration

### Environment Variables

```bash
DATABASE_URL=postgresql://localhost/mydb
OPENAI_API_KEY=sk-...
```

### Connection String Format

```
postgresql://username:password@host:port/database
```

Example:
```
postgresql://postgres:secret@localhost:5432/myapp
```

## Next Steps

- [Quick Start](quickstart.md) - Get started in 5 minutes
- [Integration Guide](integration-guide.md) - Framework integration patterns
- [API Reference](api-reference.md) - Complete API documentation
