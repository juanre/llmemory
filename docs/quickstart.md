# Quick Start

Get started with llmemory in 5 minutes.

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key (or use local embeddings)

## Installation

```bash
pip install llmemory
# Optional: install local reranker dependencies
uv add "llmemory[reranker-local]"
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

    # Search with advanced retrieval
    results = await memory.search(
        owner_id="workspace-1",
        query_text="your search query",
        search_type=SearchType.HYBRID,
        limit=5,
        query_expansion=True,   # Generate multiple query variants (optional)
        rerank=True             # Apply the configured reranker (optional)
    )
    print(f"Found {len(results)} results")

    for result in results:
        summary = result.summary or "(no summary)"
        print(f"[score={result.score:.3f}] {summary} -> {result.content[:80]}...")

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
# Retrieval tuning (optional)
LLMEMORY_ENABLE_QUERY_EXPANSION=1
LLMEMORY_ENABLE_RERANK=1
LLMEMORY_ENABLE_CHUNK_SUMMARIES=1
LLMEMORY_RERANK_MODEL=gpt-4.1-mini
LLMEMORY_RERANK_PROVIDER=openai
LLMEMORY_HNSW_PROFILE=balanced
```

Or programmatic:

```python
from llmemory import LLMemoryConfig

config = LLMemoryConfig()
config.chunking.enable_chunk_summaries = True
config.search.enable_query_expansion = True
config.search.enable_rerank = True

memory = LLMemory(connection_string="...", config=config)
```

With these toggles enabled, each ingestion will capture short summaries for downstream prompts, `LLMemory.search()` will synthesize multiple query variants before retrieval, and the reranker will refine the final hit list.
