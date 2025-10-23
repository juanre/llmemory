# Quick Start

Get up and running with llmemory in 5 minutes.

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector
- OpenAI API key (optional for local embeddings)

## Installation

```bash
# Install using uv (recommended)
uv add llmemory

# Or using pip
pip install llmemory
```

## Basic Usage

```python
import asyncio
from llmemory import LLMemory, DocumentType, SearchType

async def main():
    # Initialize
    memory = LLMemory(
        connection_string="postgresql://localhost/mydb",
        openai_api_key="sk-..."  # Or use environment variable
    )
    await memory.initialize()

    # Add a document
    result = await memory.add_document(
        owner_id="workspace-123",
        id_at_origin="user-456",
        document_name="example.txt",
        document_type=DocumentType.TEXT,
        content="Your document content here..."
    )

    # Search
    results = await memory.search_with_documents(
        owner_id="workspace-123",
        query_text="your search query",
        search_type=SearchType.HYBRID
    )

    # Clean up
    await memory.close()

# Run the example
asyncio.run(main())
```

### 3. Add documents

```python
async def run_example(memory):
    # Add a business report
    result = await memory.add_document(
        owner_id="acme-corp",
        id_at_origin="john.doe@acme.com",
        document_name="Q4-2024-Financial-Report.pdf",
        document_type=DocumentType.BUSINESS_REPORT,
        content="""
        ACME Corporation Q4 2024 Financial Report

        Executive Summary:
        - Revenue: $50M (up 25% YoY)
        - Net Profit: $12M (up 30% YoY)
        - Customer Growth: 15,000 new customers

        Key Highlights:
        1. Strong performance in cloud services division
        2. Successful launch of new AI product line
        3. Expansion into European markets showing early success

        Financial Details:
        - Cloud Services: $30M revenue (60% of total)
        - AI Products: $15M revenue (30% of total)
        - Consulting: $5M revenue (10% of total)

        Outlook for 2025:
        We expect continued growth driven by AI adoption...
        """,
        metadata={
            "fiscal_year": 2024,
            "quarter": "Q4",
            "department": "finance",
            "confidential": True
        }
    )

    print(f"✅ Added document: {result.document_id}")
    print(f"   Created {result.chunks_created} chunks")
    print(f"   Processing time: {result.processing_time_ms:.2f}ms")
```

### 4. Search documents

```python
    # Wait a moment for embeddings to generate
    await asyncio.sleep(2)

    # Semantic search
    results = await memory.search(
        owner_id="acme-corp",
        query_text="AI product revenue performance",
        search_type=SearchType.HYBRID,
        limit=5
    )

    print("\n🔍 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Content: {result.content[:150]}...")
```

### 5. List and filter documents

```python
    # List all documents
    doc_list = await memory.list_documents(
        owner_id="acme-corp",
        limit=10
    )

    print(f"\n📄 Total documents: {doc_list.total}")
    for doc in doc_list.documents:
        print(f"   - {doc.document_name} ({doc.document_type.value})")

    # Filter by metadata
    financial_docs = await memory.list_documents(
        owner_id="acme-corp",
        metadata_filter={"department": "finance"},
        limit=10
    )

    print(f"\n💰 Financial documents: {len(financial_docs.documents)}")
```

### 6. Get document details

```python
    # Get complete document with chunks
    doc_id = result.document_id
    full_doc = await memory.get_document(document_id=doc_id, include_chunks=True)

    print(f"\n📊 Document Details:")
    print(f"   Name: {full_doc.document.document_name}")
    print(f"   Type: {full_doc.document.document_type}")
    print(f"   Chunks: {len(full_doc.chunks)}")
    print(f"   Created: {full_doc.document.created_at}")
```

### 7. Get usage statistics

```python
    # Get owner statistics
        stats = await memory.get_statistics("acme-corp")

    print(f"\n📈 Usage Statistics:")
    print(f"   Documents: {stats.document_count}")
    print(f"   Chunks: {stats.chunk_count}")
    print(f"   Storage bytes: {stats.total_size_bytes}")
```

## Complete Example

Here's the full example that you can copy and run:

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

    try:
        # Add a document
        result = await memory.add_document(
            owner_id="demo-workspace",
            id_at_origin="demo@example.com",
            document_name="product-roadmap.md",
            document_type=DocumentType.MARKDOWN,
            content="""
            # Product Roadmap 2025

            ## Q1: Foundation
            - Launch new authentication system
            - Implement real-time collaboration
            - Mobile app beta release

            ## Q2: Scale
            - Multi-region deployment
            - Enterprise SSO integration
            - Advanced analytics dashboard

            ## Q3: AI Features
            - Smart document search
            - Automated insights
            - Predictive analytics

            ## Q4: Platform
            - API v2 release
            - Third-party integrations
            - Marketplace launch
            """,
            metadata={
                "year": 2025,
                "status": "draft",
                "author": "Product Team"
            }
        )

        print(f"✅ Document added successfully!")
        print(f"   ID: {result.document_id}")
        print(f"   Chunks: {result.chunks_created}")

        # Wait for embeddings
        await asyncio.sleep(2)

        # Search
        results = await memory.search_with_documents(
            owner_id="demo-workspace",
            query_text="AI and machine learning features",
            search_type=SearchType.HYBRID
        )

        print(f"\n🔍 Found {len(results.results)} relevant chunks:")
        for result in results.results:
            print(f"\n   Document: {result.document_name}")
            print(f"   Content: {result.content[:100]}...")
            print(f"   Score: {result.score:.3f}")

        # Get statistics
        stats = await memory.get_statistics("demo-workspace")
        print(f"\n📊 Workspace Statistics:")
        print(f"   Total documents: {stats.document_count}")
        print(f"   Total chunks: {stats.chunk_count}")

    finally:
        await memory.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Using Local Embeddings

To use local embeddings instead of OpenAI:

```python
# Set environment variable
export AWORD_EMBEDDING_PROVIDER=local-minilm

# Or configure in code
from llmemory import LLMemoryConfig, set_config

config = LLMemoryConfig()
config.embedding.default_provider = "local-minilm"
set_config(config)

# Then use normally
memory = LLMemory(connection_string="postgresql://localhost/mydb")
```

## Error Handling

Always handle potential errors:

```python
from llmemory import (
    LLMemoryError,
    ValidationError,
    DocumentNotFoundError
)

try:
    await memory.add_document(...)
except ValidationError as e:
    print(f"Invalid input: {e}")
except DocumentNotFoundError as e:
    print(f"Document not found: {e}")
except LLMemoryError as e:
    print(f"Memory service error: {e}")
```

## Best Practices

1. **Use context managers** for automatic cleanup:
   ```python
   async with LLMemory(connection_string="...") as memory:
       # Your code here
       pass  # Automatically closes on exit
   ```

2. **Batch operations** when possible:
   ```python
   # Delete multiple documents at once
   await memory.delete_documents_by_filter(
       owner_id="workspace",
       metadata_filter={"status": "archived"}
   )
   ```

3. **Use appropriate document types** for better chunking:
   ```python
   # Email gets different chunking than technical docs
   DocumentType.EMAIL  # Smaller chunks
   DocumentType.TECHNICAL_DOC  # Larger chunks
   ```

4. **Add meaningful metadata** for filtering:
   ```python
   additional_metadata={
       "project": "alpha",
       "version": "1.2.0",
       "tags": ["important", "reviewed"]
   }
   ```

## Next Steps

- Explore [advanced examples](../examples/)
- Read the [API Reference](api-reference.md) for all available methods
- Learn about [integration patterns](integration-guide.md)
- Set up [monitoring](monitoring.md) for production use
