# `llmemory`

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/llmemory.svg)](https://pypi.org/project/llmemory/)
[![PyPI](https://img.shields.io/pypi/v/llmemory.svg)](https://pypi.org/project/llmemory/)

A high-performance document memory system with vector search capabilities for Python applications.

## Overview

`llmemory` provides document search with:
- **Complete Document Management API** – List, retrieve, search, and manage documents without direct database access
- **State-of-the-art retrieval** using PostgreSQL with pgvector, hybrid BM25, multi-query expansion, and reranking
- **Multi-language support** with automatic language detection and normalization
- **Hierarchical chunking & summaries** with document-type specific configurations and optional auto-summaries
- **Monitoring** with Prometheus metrics and search diagnostics
- **Reliable async Postgres foundation** via [pgdbm](https://juanreyero.com/open/pgdbm/) (connection pooling, migrations, schema support)

## What's New

- 🔁 **Multi-query expansion** – Generate semantic + keyword variants automatically and fuse results with reciprocal rank fusion
- 🎯 **Configurable reranking** – Plug in OpenAI or cross-encoder rerankers (or use built-in heuristics) for higher precision on the final hit list
- 🧭 **Query routing** – Automatic answerability detection routes queries to best retrieval strategy
- 🎨 **Contextual retrieval** – Anthropic-style chunk enrichment with document context for improved semantic matching
- ⚙️ **HNSW presets** – Choose `fast`, `balanced`, or `accurate` profiles to tune pgvector index parameters and query-time `ef_search`
- 📝 **Chunk summaries** – Capture short, metadata-aware synopses during ingestion and surface them with every search hit
- 📈 **Richer diagnostics** – Search history now records query variants, latency breakdowns, rerank status, and summary usage for easy tuning

## Why `llmemory`?

Applications with document search capabilities require:

- **Vector embeddings** for semantic understanding
- **Efficient chunking** that preserves context
- **Hybrid search** combining vectors and full-text
- **Multi-tenant isolation** for SaaS applications
- **Performance optimization** for large document sets

`llmemory` provides a solution.

## Key Features

- 🚀 **Fast Search** – HNSW indexes for sub-100 ms vector searches, with multi-query expansion and optional cross-encoder reranking for harder queries
- 🌍 **Multi-language** – Automatic detection and processing for 14+ languages
- 📊 **Smart Chunking** – Document-type aware chunking with contextual enrichment, optional inline summaries, and hierarchical parent context
- 🔍 **Hybrid Search** – Combines vector and text search with reciprocal rank fusion, query routing, and rerank scores
- 📈 **Observable** – Built-in Prometheus metrics and detailed search diagnostics
- 🏢 **Multi-tenant** – Owner-based isolation for SaaS applications
- 🔌 **Flexible Embeddings** – Support for OpenAI and local embedding models

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
uv add "llmemory[reranker-local]"  # For local cross-encoder reranking
uv add "llmemory[bench]"       # For BEIR benchmarking harness

# Using pip
pip install "llmemory[monitoring]"  # For Prometheus metrics
pip install "llmemory[cache]"       # For Redis caching
pip install "llmemory[local]"       # For local embeddings
pip install "llmemory[reranker-local]"  # For cross-encoder reranking support
pip install "llmemory[bench]"       # For benchmarking harness
```

## Claude Code Skills

`llmemory` provides **expert guidance skills** for Claude Code that teach Claude how to work with the library effectively. When you use Claude Code with `llmemory`, these skills automatically activate to provide:

- ✅ Production-ready code examples
- ✅ Best practices and patterns
- ✅ Common pitfalls to avoid
- ✅ Architecture guidance
- ✅ Testing strategies

**No manual needed** - just ask Claude naturally and the right skills load automatically!

### Installation

```bash
# In Claude Code terminal, add the marketplace
/plugin marketplace add juanre/ai-tools

# Install all llmemory skills (recommended)
/plugin install llmemory@juanre-ai-tools

# Or install individual skills
/plugin install llmemory-hybrid-search@juanre-ai-tools
/plugin install llmemory-rag@juanre-ai-tools
```

### Available Skills

| Skill | Description | Install |
|-------|-------------|---------|
| `llmemory` | All llmemory skills (recommended) | `/plugin install llmemory@juanre-ai-tools` |
| `llmemory-basic-usage` | Getting started and basic operations | `/plugin install llmemory-basic-usage@juanre-ai-tools` |
| `llmemory-hybrid-search` | Vector + BM25 hybrid search | `/plugin install llmemory-hybrid-search@juanre-ai-tools` |
| `llmemory-multi-query` | Query expansion for better results | `/plugin install llmemory-multi-query@juanre-ai-tools` |
| `llmemory-multi-tenant` | Multi-tenant SaaS patterns | `/plugin install llmemory-multi-tenant@juanre-ai-tools` |
| `llmemory-rag` | Complete RAG systems | `/plugin install llmemory-rag@juanre-ai-tools` |

### How It Works

**Example: Building a RAG system**

**You ask:**
> "Build a RAG system with hybrid search for customer support docs"

**What happens:**
1. Claude sees "RAG", "hybrid search"
2. Automatically loads `llmemory-hybrid-search` and `llmemory-rag` skills
3. Guides you through document ingestion, search setup, and retrieval
4. Shows you how to combine vector + keyword search

**Result:** Complete RAG implementation with best practices built-in!

**Example: Multi-tenant document search**

**You ask:**
> "Add document search with owner-based isolation for my SaaS app"

**What happens:**
1. Claude sees "SaaS", "owner-based isolation"
2. Loads `llmemory-multi-tenant` skill
3. Provides expert guidance on multi-tenant patterns
4. Shows you exactly how to implement owner isolation

**Result:** Production-ready multi-tenant search with proper data isolation!

## Documentation

- 📖 [Installation Guide](docs/installation.md) - Detailed setup instructions
- 🚀 [Quick Start](docs/quickstart.md) - Get running in 5 minutes
- 🎯 [Usage Patterns](docs/usage-patterns.md) - Standalone vs shared pool patterns
- 📚 [API Reference](docs/api-reference.md) - Complete API documentation
- 🔧 [Integration Guide](docs/integration-guide.md) - Framework integration patterns
- 🗄️ [Migration Guide](docs/migrations.md) - How migrations work in each pattern
- 📊 [Monitoring Guide](docs/monitoring.md) - Production monitoring setup
- 💡 [Examples](examples/) - Working examples for common use cases
- 🧪 `bench/beir_runner.py` - BEIR benchmarking harness (requires `llmemory[bench]`)

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
