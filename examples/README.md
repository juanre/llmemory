# llmemory Examples

This directory contains examples demonstrating how to use llmemory in different scenarios.

## Core Examples

### Usage Patterns
These examples show the three main deployment patterns for llmemory:

- **[pattern_1_standalone.py](pattern_1_standalone.py)** - Standalone usage where llmemory manages its own database connection
- **[pattern_2_library.py](pattern_2_library.py)** - Building a library that uses llmemory internally
- **[pattern_3_application.py](pattern_3_application.py)** - Production application using shared connection pools

### Feature Examples

- **[local_embeddings_example.py](local_embeddings_example.py)** - Using local embedding models instead of OpenAI for privacy/cost
- **[validation_and_config_example.py](validation_and_config_example.py)** - Configuration management and error handling
- **[production_integration.py](production_integration.py)** - Production deployment with monitoring and best practices

## Quick Start

The simplest way to get started:

```python
from llmemory import LLMemory, DocumentType, SearchType

# Initialize
memory = LLMemory(
    connection_string="postgresql://localhost/mydb",
    openai_api_key="sk-..."
)
await memory.initialize()

# Add a document
result = await memory.add_document(
    owner_id="workspace-123",
    id_at_origin="user-456",
    document_name="example.txt",
    document_type=DocumentType.GENERAL,
    content="Your content here..."
)

# Search
results = await memory.search_with_documents(
    owner_id="workspace-123",
    query_text="your search query",
    search_type=SearchType.HYBRID
)
```

## Running the Examples

1. Set up your environment:
```bash
export DATABASE_URL="postgresql://localhost/memory_demo"
export OPENAI_API_KEY="sk-..."  # Optional for local embeddings
```

2. Run an example:
```bash
python examples/pattern_1_standalone.py
```

## Which Example Should I Use?

- **New to llmemory?** Start with `pattern_1_standalone.py`
- **Building a library?** See `pattern_2_library.py`
- **Production application?** Check `pattern_3_application.py`
- **Want local embeddings?** Try `local_embeddings_example.py`
- **Need advanced config?** Look at `validation_and_config_example.py`
