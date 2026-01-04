"""Tests for the MemoryManager."""

import json

from datetime import datetime

import pytest

from llmemory.manager import MemoryManager
from llmemory.models import DocumentType, SearchQuery, SearchType


@pytest.mark.asyncio
class TestMemoryManager:
    """Test MemoryManager operations."""

    async def test_manager_initialization(self, memory_db):
        """Test manager initialization with existing db."""
        # Manager should be created using the create class method
        manager = await MemoryManager.create(memory_db.db.config.get_dsn())
        assert manager.db is not None
        await manager.db.close()

    async def test_add_document(self, memory_manager):
        """Test adding a document."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test_document.pdf",
            document_type=DocumentType.PDF,
            document_date=datetime.now(),
            metadata={"source": "upload", "size": 1024},
        )

        assert doc.id_at_origin == "user123"
        assert doc.document_name == "test_document.pdf"
        assert doc.document_type == DocumentType.PDF
        assert doc.metadata["source"] == "upload"
        assert doc.created_at is not None

    async def test_add_chunks(self, memory_manager):
        """Test adding chunks to a document."""
        # First add a document
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.md",
            document_type=DocumentType.MARKDOWN,
        )

        # Add chunks
        chunks_data = [
            ("# Introduction\nThis is the introduction.", {"section": "intro"}),
            ("## Background\nSome background information.", {"section": "background"}),
            ("## Methods\nThe methods used.", {"section": "methods"}),
        ]

        chunks = await memory_manager.add_chunks(document_id=doc.document_id, chunks=chunks_data)

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk.content == chunks_data[i][0]
            assert chunk.metadata == chunks_data[i][1]
            assert chunk.document_id == doc.document_id

    async def test_add_chunks_strips_nul_bytes(self, memory_manager):
        """Ensure NUL bytes are stripped before chunks are stored."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="nul-test.txt",
            document_type=DocumentType.TEXT,
        )

        chunks = await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[("Hello\x00World", {"note": "A\x00B"})],
        )

        assert chunks[0].content == "HelloWorld"
        assert "\x00" not in chunks[0].content
        assert chunks[0].metadata["note"] == "AB"

        query = memory_manager.db.db_manager.prepare_query(
            """
        SELECT content, metadata
        FROM {{tables.document_chunks}}
        WHERE document_id = $1
        ORDER BY chunk_index
        LIMIT 1
        """
        )
        row = await memory_manager.db.db_manager.fetch_one(query, str(doc.document_id))
        assert row["content"] == "HelloWorld"
        assert "\x00" not in row["content"]
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert metadata["note"] == "AB"

    async def test_hierarchical_chunks(self, memory_manager):
        """Test adding hierarchical chunks."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.md",
            document_type=DocumentType.MARKDOWN,
        )

        # Add parent chunk
        parent_chunks = await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[("# Chapter 1\nThis is a complete chapter.", {"level": "chapter"})],
            chunk_level=1,
        )
        parent_chunk = parent_chunks[0]

        # Add child chunks
        child_chunks = await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[
                ("This is the first part of the chapter.", {"level": "section"}),
                ("This is the second part of the chapter.", {"level": "section"}),
            ],
            parent_chunk_id=parent_chunk.chunk_id,
            chunk_level=0,
        )
        child1 = child_chunks[0]
        child2 = child_chunks[1]

        assert child1.parent_chunk_id == parent_chunk.chunk_id
        assert child2.parent_chunk_id == parent_chunk.chunk_id
        assert parent_chunk.chunk_level == 1
        assert child1.chunk_level == 0
        assert child2.chunk_level == 0

    async def test_deduplication(self, memory_manager):
        """Test chunk deduplication."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        # Add the same content twice
        content = "This is duplicate content that should be deduplicated."

        chunks1 = await memory_manager.add_chunks(
            document_id=doc.document_id, chunks=[(content, {"attempt": 1})]
        )
        chunk1 = chunks1[0]

        chunks2 = await memory_manager.add_chunks(
            document_id=doc.document_id, chunks=[(content, {"attempt": 2})]
        )
        chunk2 = chunks2[0]

        # Should return the same chunk (deduplication)
        assert chunk1.chunk_id == chunk2.chunk_id
        assert chunk1.content_hash == chunk2.content_hash

    async def test_update_chunk_embedding(self, memory_manager):
        """Test updating chunk embeddings."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        chunks = await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[("This is test content for embeddings.", {})],
        )
        chunk = chunks[0]

        # Update with fake embedding
        embedding = [0.1] * 1536  # text-embedding-3-small dimension

        await memory_manager.update_chunk_embedding(
            chunk_id=chunk.chunk_id,
            embedding=embedding,
            provider_id="openai-text-embedding-3-small",
        )

        # Verify embedding was stored (would need direct DB check)
        # For now, just verify the method doesn't raise

    async def test_search_without_embeddings(self, memory_manager):
        """Test text search without embeddings."""
        # Add test documents
        docs_data = [
            ("Python programming language", "python.txt"),
            ("JavaScript web development", "js.txt"),
            ("Machine learning with Python", "ml.txt"),
        ]

        for content, name in docs_data:
            doc = await memory_manager.add_document(
                owner_id="workspace_test",
                id_at_origin="user123",
                document_name=name,
                document_type=DocumentType.TEXT,
            )

            await memory_manager.add_chunks(document_id=doc.document_id, chunks=[(content, {})])

        # Search for Python
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="Python",
            search_type=SearchType.TEXT,
            limit=5,
        )

        results = await memory_manager.search(query)

        assert len(results) > 0
        # Should find Python-related documents
        python_found = any("python" in r.content.lower() for r in results)
        assert python_found

    async def test_search_with_embeddings(self, memory_manager):
        """Test vector search with embeddings."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        chunks = await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[("Artificial intelligence and machine learning", {})],
        )
        chunk = chunks[0]

        # Add fake embedding
        embedding = [0.1] * 1536
        await memory_manager.update_chunk_embedding(chunk_id=chunk.chunk_id, embedding=embedding)

        # Search with fake query embedding
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="AI and ML",
            search_type=SearchType.VECTOR,
            limit=5,
        )

        results = await memory_manager.search(query, query_embedding=embedding)

        # Should find our chunk (with perfect match since same embedding)
        assert len(results) > 0
        assert results[0].chunk_id == chunk.chunk_id

    async def test_hybrid_search(self, memory_manager):
        """Test hybrid search combining text and vector."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        chunks_data = [
            ("Python is great for machine learning", [0.1] * 1536),
            ("JavaScript for web development", [0.2] * 1536),
            ("Machine learning algorithms in Python", [0.15] * 1536),
        ]

        for content, embedding in chunks_data:
            chunks = await memory_manager.add_chunks(
                document_id=doc.document_id, chunks=[(content, {})]
            )
            chunk = chunks[0]

            await memory_manager.update_chunk_embedding(
                chunk_id=chunk.chunk_id, embedding=embedding
            )

        # Hybrid search for "Python machine learning"
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="Python machine learning",
            search_type=SearchType.HYBRID,
            limit=5,
            alpha=0.5,  # Equal weight
        )

        # Use a query embedding similar to first chunk
        query_embedding = [0.12] * 1536

        results = await memory_manager.search(query, query_embedding=query_embedding)

        assert len(results) > 0
        # Should rank Python ML content higher
        assert "python" in results[0].content.lower()
        assert "machine learning" in results[0].content.lower()

    async def test_search_with_metadata_filter(self, memory_manager):
        """Test search with metadata filtering."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        # Add chunks with different metadata
        await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[
                ("Python programming", {"language": "python", "topic": "programming"}),
                (
                    "JavaScript programming",
                    {"language": "javascript", "topic": "programming"},
                ),
            ],
        )

        # Search with metadata filter
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="programming",
            search_type=SearchType.TEXT,
            metadata_filter={"language": "python"},
            limit=5,
        )

        results = await memory_manager.search(query)

        assert len(results) == 1
        # Handle metadata as dict or JSON string
        metadata = results[0].metadata
        if isinstance(metadata, str):
            import json

            metadata = json.loads(metadata)
        assert metadata["language"] == "python"

    async def test_process_document(self, memory_manager):
        """Test complete document processing."""
        content = """
# Introduction to Python

Python is a high-level programming language.

## Features

Python has many great features:
- Easy to learn
- Powerful libraries
- Great community

## Applications

Python is used in:
- Web development
- Data science
- Machine learning
"""

        doc, chunks = await memory_manager.process_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="python_intro.md",
            document_type=DocumentType.MARKDOWN,
            content=content,
            chunking_strategy="hierarchical",
        )

        assert doc.document_name == "python_intro.md"
        assert len(chunks) > 0

        # Should have both parent and child chunks
        parent_chunks = [c for c in chunks if c.chunk_level > 1]
        child_chunks = [c for c in chunks if c.chunk_level <= 1]

        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0

    async def test_get_document_chunks(self, memory_manager):
        """Test retrieving chunks for a document."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        # Add multiple chunks
        chunks_data = []
        for i in range(3):
            chunks_data.append((f"Chunk {i} content", {"index": i}))

        # Add all chunks at once
        chunks = await memory_manager.add_chunks(document_id=doc.document_id, chunks=chunks_data)

        assert len(chunks) == 3
        # Should have created them in order
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["index"] == i

    async def test_get_pending_embeddings(self, memory_manager):
        """Test getting chunks pending embedding generation."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        # Add chunks (they'll be queued for embeddings)
        chunks_data = []
        for i in range(3):
            chunks_data.append((f"Content {i} for embedding", {}))

        chunks = await memory_manager.add_chunks(document_id=doc.document_id, chunks=chunks_data)

        # Get pending embeddings
        pending = await memory_manager.get_pending_embeddings(limit=10)

        # Should have our chunks pending
        assert len(pending) >= 3
        pending_chunk_ids = [p.chunk_id for p in pending]
        for chunk in chunks:
            assert chunk.chunk_id in pending_chunk_ids

    async def test_delete_document(self, memory_manager):
        """Test document deletion cascade."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="to_delete.txt",
            document_type=DocumentType.TEXT,
        )

        # Add chunks
        chunks = await memory_manager.add_chunks(
            document_id=doc.document_id, chunks=[("This will be deleted", {})]
        )
        chunk = chunks[0]

        # Delete document
        await memory_manager.delete_document(doc.document_id)

        # Chunks should also be deleted (would need to verify with direct DB check)
        # For now, verify search doesn't find it
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="deleted",
            search_type=SearchType.TEXT,
            limit=5,
        )

        results = await memory_manager.search(query)

        # Should not find the deleted content
        deleted_found = any(r.chunk_id == chunk.chunk_id for r in results)
        assert not deleted_found

    async def test_search_logging(self, memory_manager):
        """Test that searches are logged."""
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="user123",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        await memory_manager.add_chunks(
            document_id=doc.document_id,
            chunks=[("Test content for search logging", {})],
        )

        # Perform search
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="search logging",
            search_type=SearchType.TEXT,
            limit=5,
        )

        results = await memory_manager.search(query)

        # Search should be logged (would need to verify with direct DB check)
        # For now, just verify search completed
        assert isinstance(results, list)
