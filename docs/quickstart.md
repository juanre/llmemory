# Quick Start

Get started with llmemory in 5 minutes.

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key (or use local embeddings)

## Installation

```bash
pip install llmemory
```

## Basic Example

```python
import asyncio
from llmemory import LLMemory, DocumentType, SearchType

async def main():
    # Initialize
    memory = LLMemory(
        connection_string="postgresql://localhost/mydb",
        openai_api_key="sk-..."
    )
    await memory.initialize()

    # Add a document
    result = await memory.add_document(
        owner_id="workspace-1",
        id_at_origin="user-123",
        document_name="example.txt",
        document_type=DocumentType.TEXT,
        content="Your document content here...",
        metadata={"category": "example"}
    )
    print(f"Created document with {result.chunks_created} chunks")

    # Search
    results = await memory.search(
        owner_id="workspace-1",
        query_text="your search query",
        search_type=SearchType.HYBRID,
        limit=10
    )
    print(f"Found {len(results)} results")

    for result in results:
        print(f"Score {result.score:.3f}: {result.content[:100]}...")

    # Clean up
    await memory.close()

asyncio.run(main())
```

## Next Steps

- **[API Reference](api-reference.md)** - Complete API documentation
- **[Integration Guide](integration-guide.md)** - Framework integration and deployment patterns
- **[Testing Guide](testing-guide.md)** - Testing with llmemory

## Common Patterns

### List Documents

```python
docs = await memory.list_documents(
    owner_id="workspace-1",
    limit=20,
    offset=0
)
```

### Get Document with Chunks

```python
doc = await memory.get_document(
    owner_id="workspace-1",
    document_id="uuid-here",
    include_chunks=True
)
```

### Delete Documents

```python
result = await memory.delete_documents(
    owner_id="workspace-1",
    document_ids=["uuid-1", "uuid-2"]
)
```

### Get Statistics

```python
stats = await memory.get_statistics("workspace-1")
print(f"Documents: {stats.document_count}, Chunks: {stats.chunk_count}")
```

## Configuration

Environment variables:

```bash
DATABASE_URL=postgresql://localhost/mydb
OPENAI_API_KEY=sk-...
```

Or programmatic:

```python
from llmemory import LLMemoryConfig

config = LLMemoryConfig()
config.chunking.chunk_size = 1500

memory = LLMemory(connection_string="...", config=config)
```
