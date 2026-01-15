# Testing Guide for Memory Service

## Using Memory Service Test Fixtures

The memory service provides reusable test fixtures that can be imported into your own test suites. This allows you to easily test code that depends on the memory service without having to recreate common test setups.

### Installation

First, ensure you have the test dependencies installed:

```bash
uv add llmemory pytest pytest-asyncio
```

### Basic Setup

In your test file or `conftest.py`, import the fixtures:

```python
# conftest.py
from pgdbm.fixtures.conftest import *  # Required for test_db_factory
from llmemory.testing import *       # Memory service fixtures
```

### Available Fixtures

#### Database Fixtures

- **`memory_db`**: Creates an initialized MemoryDatabase instance with migrations applied
- **`memory_manager`**: Creates a MemoryManager instance connected to a test database

#### OpenAI Fixtures

- **`openai_client`**: Creates an async OpenAI client (requires OPENAI_API_KEY environment variable)
- **`sample_embeddings`**: Provides pre-generated embeddings for testing
- **`create_embedding`**: Factory function to generate embeddings for any text

#### LLMemory Fixtures

- **`memory_library`**: Creates a fully initialized LLMemory instance
- **`memory_library_with_embeddings`**: LLMemory instance with pre-populated documents and embeddings

#### Test Data Fixtures

- **`sample_documents`**: Provides sample document content for testing

### Example Usage

```python
import pytest
from llmemory.testing import *
from pgdbm.fixtures.conftest import *

@pytest.mark.asyncio
async def test_my_memory_integration(memory_library):
    """Test my application's integration with memory service."""
    # Add a document
    result = await memory_library.add_document(
        owner_id="test_user",
        id_at_origin="doc1",
        document_name="test.txt",
        document_type="text",
        content="This is a test document"
    )

    # Search for it
    results = await memory_library.search(
        owner_id="test_user",
        query_text="test document",
        search_type="text"
    )

    assert len(results) > 0
    assert results[0].document_id == result.document.document_id

@pytest.mark.asyncio
async def test_with_embeddings(memory_library_with_embeddings):
    """Test with pre-populated documents and embeddings."""
    # The fixture provides documents with embeddings already generated
    results = await memory_library_with_embeddings.search(
        owner_id="test_workspace",
        query_text="machine learning",
        search_type="vector"
    )

    assert len(results) > 0
    # Should find the AI/ML document
    assert any("ai_ml_intro" in r.document_name for r in results)

@pytest.mark.asyncio
async def test_custom_embeddings(create_embedding):
    """Test creating custom embeddings."""
    embedding = await create_embedding("Custom text for embedding")
    assert len(embedding) == 1536  # OpenAI text-embedding-3-small dimension
```

### Environment Variables

Some fixtures require environment variables:

- `OPENAI_API_KEY`: Required for OpenAI-based fixtures (`openai_client`, `sample_embeddings`, `create_embedding`)
- Tests requiring OpenAI will be skipped if this is not set

### Advanced Usage

You can also use the lower-level fixtures for more control:

```python
@pytest.mark.asyncio
async def test_direct_db_access(memory_db):
    """Test using the database directly."""
    # Insert a document
    doc_id = await memory_db.insert_document(
        owner_id="test",
        id_at_origin="origin1",
        document_type="text",
        document_name="test.txt"
    )

    # Check it exists
    exists = await memory_db.document_exists(doc_id)
    assert exists

@pytest.mark.asyncio
async def test_with_manager(memory_manager):
    """Test using the memory manager."""
    # Add document with automatic chunking
    doc = await memory_manager.add_document(
        owner_id="test",
        id_at_origin="origin1",
        document_name="test.txt",
        document_type="text",
        metadata={"custom": "data"}
    )

    # Add chunks
    chunks = await memory_manager.add_chunks(
        document_id=doc.document_id,
        chunks=[
            {"content": "First chunk", "metadata": {}},
            {"content": "Second chunk", "metadata": {}}
        ]
    )

    assert len(chunks) == 2
```

### Creating Your Own Fixtures

You can build on these fixtures to create your own:

```python
@pytest_asyncio.fixture
async def my_app_memory(memory_library):
    """Create a memory instance configured for my app."""
    # Add some app-specific documents
    await memory_library.add_document(
        owner_id="my_app",
        id_at_origin="config",
        document_name="app_config.json",
        document_type="json",
        content='{"version": "1.0", "features": ["search", "embed"]}'
    )

    yield memory_library

@pytest.mark.asyncio
async def test_my_app(my_app_memory):
    """Test with app-specific setup."""
    results = await my_app_memory.search(
        owner_id="my_app",
        query_text="features",
        search_type="text"
    )
    assert len(results) > 0
```

### Tips

1. **Isolation**: Each test gets its own database schema, ensuring tests don't interfere with each other
2. **Performance**: The fixtures handle all initialization and cleanup automatically
3. **Skipping**: Tests requiring OpenAI will automatically skip if the API key is not available
4. **Debugging**: Use the `memory_db` fixture for direct database access when debugging issues

### Troubleshooting

If you encounter import errors:

1. Ensure you have installed the test dependencies
2. Make sure `pgdbm` fixtures are imported before `llmemory` fixtures
3. Check that your Python path includes the memory service package

For more examples, see the memory service test suite in the `tests/` directory.

## Test Status

Test results are **environment-dependent**:

- Some tests and fixtures **skip automatically** when optional dependencies or credentials are missing (for example: `OPENAI_API_KEY`, `sentence-transformers`, or external corpora under `tests/res/`).
- Some suites are intentionally marked `@pytest.mark.skip(...)` in-repo because they are performance/quality regression checks that require a specific environment and can be flaky in generic CI runners (for example: `tests/test_search_quality.py` and `tests/test_search_performance.py`).

If you want to run everything locally, start with:

```bash
uv run pytest
```

Then, selectively enable the heavier suites by removing/adjusting the `skip` markers in the relevant test modules.

## Optional Dependencies

The memory service supports optional features through extra dependencies:

### Local Embeddings
For local embedding models (sentence-transformers):
```bash
uv add "llmemory[local]"
```

This installs:
- sentence-transformers
- torch

### Monitoring
For Prometheus metrics:
```bash
uv add "llmemory[monitoring]"
```

### All Optional Features
```bash
uv add "llmemory[local,monitoring]"
```

## Lazy Loading

The codebase implements lazy loading for expensive dependencies:

1. **OpenAI Client**: Loaded only when first embedding is generated
2. **Local Models**: sentence-transformers and torch loaded only when local provider is used
3. **Prometheus Metrics**: Initialized only if monitoring is enabled

This ensures:
- Fast startup times
- Minimal memory usage when features aren't used
- No unnecessary dependencies loaded
