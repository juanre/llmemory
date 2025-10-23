# llmemory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/llmemory.svg)](https://pypi.org/project/llmemory/)
[![PyPI](https://img.shields.io/pypi/v/llmemory.svg)](https://pypi.org/project/llmemory/)

A high-performance document memory system with vector search capabilities for Python applications.

## Overview

llmemory provides intelligent document processing with:
- **Complete Document Management API** - List, retrieve, search, and manage documents without database access
- **State-of-the-art vector search** using PostgreSQL with pgvector and HNSW indexing
- **Multi-language support** with automatic language detection and normalization
- **Hybrid search** combining semantic and full-text search with Reciprocal Rank Fusion
- **Hierarchical chunking** with document-type specific configurations
- **Production-ready monitoring** with Prometheus metrics and health checks

## What's New in v0.2.0

- 📋 **Document Listing** - Browse and filter documents with pagination
- 📄 **Full Document Retrieval** - Get documents with chunks and metadata
- 🔍 **Enhanced Search** - Search results now include document metadata
- 📊 **Statistics API** - Get usage statistics and document analytics
- 🗑️ **Batch Operations** - Delete multiple documents by ID or metadata
- 🎯 **Better Return Types** - Detailed results from all operations

## Why llmemory?

Building applications with document search capabilities requires solving complex problems:

- **Vector embeddings** for semantic understanding
- **Efficient chunking** that preserves context
- **Hybrid search** combining vectors and full-text
- **Multi-tenant isolation** for SaaS applications
- **Performance optimization** for large document sets

llmemory provides a production-ready solution for these challenges.

## Key Features

- 🚀 **Fast Search**: HNSW indexes for sub-100ms vector searches
- 🌍 **Multi-language**: Automatic detection and processing for 14+ languages
- 📊 **Smart Chunking**: Document-type aware chunking (emails, code, reports, etc.)
- 🔍 **Hybrid Search**: Combines vector and text search for best results
- 📈 **Observable**: Built-in Prometheus metrics and performance tracking
- 🏢 **Multi-tenant**: Owner-based isolation for SaaS applications
- 🔌 **Flexible Embeddings**: Support for OpenAI and local embedding models

## Quick Start

```python
from llmemory import LLMemory, DocumentType, SearchType

# Initialize
memory = LLMemory(
    connection_string="postgresql://localhost/mydb",
    openai_api_key="sk-..."
)
await memory.initialize()

# Add a document - returns detailed results
result = await memory.add_document(
    owner_id="workspace-123",
    id_at_origin="user-456",
    document_name="project-report.pdf",
    document_type=DocumentType.REPORT,
    content="Your document content here..."
)
print(f"Created {result.chunks_created} chunks in {result.processing_time_ms}ms")

# List documents with filtering
docs = await memory.list_documents(
    owner_id="workspace-123",
    document_type=DocumentType.REPORT,
    metadata_filter={"status": "active"}
)

# Search with document metadata
results = await memory.search_with_documents(
    owner_id="workspace-123",
    query_text="project timeline",
    search_type=SearchType.HYBRID
)
for result in results.results:
    print(f"Found in: {result.document_name} - {result.content[:100]}...")

# Get statistics
stats = await memory.get_statistics("workspace-123")
print(f"Total: {stats.document_count} docs, {stats.chunk_count} chunks")
```

## Installation

```bash
# Install using uv (recommended)
uv add llmemory

# Or using pip
pip install llmemory
```

Or with optional dependencies:

```bash
# Using uv
uv add "llmemory[monitoring]"  # For Prometheus metrics
uv add "llmemory[cache]"       # For Redis caching
uv add "llmemory[local]"       # For local embeddings

# Using pip
pip install "llmemory[monitoring]"  # For Prometheus metrics
pip install "llmemory[cache]"       # For Redis caching
pip install "llmemory[local]"       # For local embeddings
```

## Documentation

- 📖 [Installation Guide](docs/installation.md) - Detailed setup instructions
- 🚀 [Quick Start](docs/quickstart.md) - Get running in 5 minutes
- 🎯 [Usage Patterns](docs/usage-patterns.md) - Standalone vs shared pool patterns
- 📚 [API Reference](docs/api-reference.md) - Complete API documentation
- 🔧 [Integration Guide](docs/integration-guide.md) - Framework integration patterns
- 🗄️ [Migration Guide](docs/migrations.md) - How migrations work in each pattern
- 📊 [Monitoring Guide](docs/monitoring.md) - Production monitoring setup
- 💡 [Examples](examples/) - Working examples for common use cases

## Performance

- **Search latency**: < 100ms (p95) with proper indexing
- **Throughput**: 1000+ searches/second with caching
- **Document processing**: Handles documents up to 1MB efficiently
- **Multi-language**: Processes 14+ languages with automatic detection

## Requirements

- PostgreSQL 14+ with pgvector extension
- Python 3.10+
- OpenAI API key (or local embedding models)

## License

MIT License - see LICENSE file for details.
