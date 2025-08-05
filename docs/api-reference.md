# API Reference

## Overview

The llmemory library provides a high-performance document memory system with vector search capabilities. The main interface is the `AwordMemory` class, which handles document storage, chunking, embedding generation, and semantic search.

## Core Classes

### AwordMemory

The main interface for interacting with the memory service.

```python
from llmemory import AwordMemory

memory = AwordMemory(
    connection_string="postgresql://user:pass@localhost/db",
    openai_api_key="sk-...",  # Optional if using local embeddings
    schema="myapp"  # Optional schema isolation
)
```

#### Methods

##### `async def initialize()`
Initialize the database connection and apply migrations.

```python
await memory.initialize()
```

##### `async def add_document(...) -> DocumentAddResult`
Add a document to the memory system.

**Parameters:**
- `owner_id` (str): Workspace or tenant identifier
- `id_at_origin` (str): User identifier who created the document
- `document_name` (str): Name of the document
- `document_type` (DocumentType): Type of document (EMAIL, REPORT, etc.)
- `content` (str): Document content
- `additional_metadata` (dict, optional): Extra metadata

**Returns:** `DocumentAddResult` with processing details

```python
result = await memory.add_document(
    owner_id="workspace-123",
    id_at_origin="user-456",
    document_name="quarterly-report.pdf",
    document_type=DocumentType.REPORT,
    content="Q3 2024 Financial Report..."
)
```

##### `async def list_documents(...) -> DocumentListResult`
List documents with optional filtering.

**Parameters:**
- `owner_id` (str): Workspace identifier
- `document_type` (DocumentType, optional): Filter by type
- `metadata_filter` (dict, optional): Filter by metadata
- `offset` (int): Pagination offset (default: 0)
- `limit` (int): Results per page (default: 20)

**Returns:** `DocumentListResult` with documents and pagination info

```python
docs = await memory.list_documents(
    owner_id="workspace-123",
    document_type=DocumentType.REPORT,
    limit=10
)
```

##### `async def get_document(...) -> DocumentWithChunks`
Get a complete document with all its chunks.

**Parameters:**
- `owner_id` (str): Workspace identifier
- `document_id` (str): Document UUID

**Returns:** `DocumentWithChunks` containing document info and chunks

```python
doc = await memory.get_document(
    owner_id="workspace-123",
    document_id="550e8400-e29b-41d4-a716-446655440000"
)
```

##### `async def search(...) -> List[SearchResult]`
Search for relevant document chunks.

**Parameters:**
- `owner_id` (str): Workspace identifier
- `query_text` (str): Search query
- `search_type` (SearchType): VECTOR, TEXT, or HYBRID
- `limit` (int): Max results (default: 10)
- `metadata_filter` (dict, optional): Filter results

**Returns:** List of `SearchResult` objects

```python
results = await memory.search(
    owner_id="workspace-123",
    query_text="revenue growth strategies",
    search_type=SearchType.HYBRID
)
```

##### `async def search_with_documents(...) -> SearchResultWithDocuments`
Search with document metadata included.

**Parameters:** Same as `search()`

**Returns:** `SearchResultWithDocuments` with enriched results

```python
results = await memory.search_with_documents(
    owner_id="workspace-123",
    query_text="customer feedback",
    search_type=SearchType.HYBRID
)
```

##### `async def delete_document(...) -> DeleteResult`
Delete a single document and its chunks.

**Parameters:**
- `owner_id` (str): Workspace identifier
- `document_id` (str): Document UUID

**Returns:** `DeleteResult` with deletion details

```python
result = await memory.delete_document(
    owner_id="workspace-123",
    document_id="550e8400-e29b-41d4-a716-446655440000"
)
```

##### `async def delete_documents_by_filter(...) -> DeleteResult`
Delete multiple documents by criteria.

**Parameters:**
- `owner_id` (str): Workspace identifier
- `document_ids` (List[str], optional): Specific document IDs
- `metadata_filter` (dict, optional): Filter by metadata

**Returns:** `DeleteResult` with deletion details

```python
result = await memory.delete_documents_by_filter(
    owner_id="workspace-123",
    metadata_filter={"status": "archived"}
)
```

##### `async def get_statistics(...) -> OwnerStatistics`
Get usage statistics for an owner.

**Parameters:**
- `owner_id` (str): Workspace identifier

**Returns:** `OwnerStatistics` with counts and usage info

```python
stats = await memory.get_statistics("workspace-123")
print(f"Documents: {stats.document_count}")
print(f"Storage: {stats.total_size_mb:.2f} MB")
```

## Data Models

### DocumentType
Enum for supported document types:
- `EMAIL` - Email messages
- `BUSINESS_REPORT` - Business reports
- `TECHNICAL_DOC` - Technical documentation
- `PRESENTATION` - Presentations
- `LEGAL_DOCUMENT` - Legal documents
- `PDF` - PDF documents
- `MARKDOWN` - Markdown files
- `CODE` - Source code
- `GENERAL` - General documents

### SearchType
Enum for search methods:
- `VECTOR` - Semantic vector search only
- `TEXT` - Full-text search only
- `HYBRID` - Combined vector and text search (recommended)

### SearchResult
Individual search result:
```python
@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    chunk_level: int
    parent_chunk_id: Optional[str]
    metadata: Dict[str, Any]
```

### DocumentAddResult
Result of adding a document:
```python
@dataclass
class DocumentAddResult:
    document_id: str
    chunks_created: int
    embeddings_queued: int
    processing_time_ms: float
    document_name: str
```

### DocumentListResult
Paginated document list:
```python
@dataclass
class DocumentListResult:
    documents: List[Document]
    total_count: int
    offset: int
    limit: int
    has_more: bool
```

### OwnerStatistics
Usage statistics:
```python
@dataclass
class OwnerStatistics:
    owner_id: str
    document_count: int
    chunk_count: int
    total_embeddings: int
    total_size_mb: float
    documents_by_type: Dict[str, int]
    created_at: datetime
```

## Configuration

### Environment Variables

Configure the service through environment variables:

```bash
# Embedding Provider Settings
AWORD_EMBEDDING_PROVIDER=openai  # or "local-minilm"
AWORD_OPENAI_API_KEY=sk-...
AWORD_OPENAI_MODEL=text-embedding-3-small

# Local Model Settings (if using local embeddings)
AWORD_LOCAL_MODEL=all-MiniLM-L6-v2
AWORD_LOCAL_DEVICE=cuda  # or "cpu"
AWORD_LOCAL_CACHE_DIR=/path/to/models

# Database Settings
AWORD_DB_MAX_POOL_SIZE=20

# Search Settings
AWORD_SEARCH_CACHE_TTL=3600

# Logging
AWORD_LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from llmemory import AwordMemoryConfig, set_config

config = AwordMemoryConfig()
config.embedding.default_provider = "local-minilm"
config.search.default_limit = 20
config.database.max_pool_size = 30

set_config(config)
```

## Error Handling

The library defines specific exceptions for different error scenarios:

```python
from llmemory import (
    AwordMemoryError,  # Base exception
    ValidationError,   # Invalid input
    DocumentNotFoundError,  # Document doesn't exist
    EmbeddingError,    # Embedding generation failed
    SearchError,       # Search operation failed
    ConnectionError,   # Database connection issues
)

try:
    await memory.add_document(...)
except ValidationError as e:
    print(f"Invalid input: {e}")
except AwordMemoryError as e:
    print(f"Memory service error: {e}")
```

## Advanced Usage

### Connection Pool Sharing

Share connection pools across multiple instances:

```python
# Create shared pool
pool = await memory.create_shared_pool()

# Multiple services share the pool
service1 = AwordMemory(pool=pool, schema="service1")
service2 = AwordMemory(pool=pool, schema="service2")
```

### Custom Embedding Providers

Configure multiple embedding providers:

```python
config = AwordMemoryConfig()

# Add a custom provider
config.embedding.providers["custom"] = EmbeddingProviderConfig(
    provider_type="openai",
    model_name="text-embedding-3-large",
    dimension=3072,
    api_key="sk-..."
)

# Set as default
config.embedding.default_provider = "custom"

set_config(config)
```

### Monitoring

Access performance metrics:

```python
# Enable monitoring
memory = AwordMemory(
    connection_string="...",
    enable_monitoring=True
)

# Get metrics (if prometheus-client installed)
from prometheus_client import generate_latest
metrics = generate_latest()
```

## Best Practices

1. **Connection Management**: Always call `initialize()` before operations and `close()` when done
2. **Batch Operations**: Use `delete_documents_by_filter()` for bulk deletions
3. **Search Type**: Use `HYBRID` search for best results in most cases
4. **Metadata**: Store searchable attributes in metadata for filtering
5. **Owner IDs**: Use consistent workspace/tenant IDs for data isolation
6. **Error Handling**: Always handle `AwordMemoryError` exceptions
7. **Resource Cleanup**: Use async context managers when possible

```python
# Recommended pattern
async with AwordMemory(connection_string="...") as memory:
    await memory.add_document(...)
    results = await memory.search(...)
```
