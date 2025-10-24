"""Tests for llmemory database integration using high-level APIs."""

from datetime import datetime
from uuid import uuid4

import pytest
from llmemory.models import DocumentType


@pytest.mark.asyncio
class TestDatabaseSetup:
    """Test database setup and pgvector integration."""

    async def test_pgvector_extension_enabled(self, memory_db):
        """Test that pgvector extension is properly enabled."""
        # Use high-level test method
        vector_ops_work = await memory_db.test_vector_operations()
        assert vector_ops_work is True

    async def test_database_connection(self, memory_db):
        """Test that database connection is working."""
        # Test by attempting to get providers
        providers = await memory_db.get_embedding_providers()
        assert isinstance(providers, list)

    async def test_tables_created(self, memory_db):
        """Test that all required tables are created via operations."""
        # Try operations that would fail if tables don't exist

        # Documents table - count documents
        count = await memory_db.count_documents("test_owner")
        assert count >= 0

        # Chunks table - count chunks
        count = await memory_db.count_chunks()
        assert count >= 0

        # Embedding providers - get providers
        providers = await memory_db.get_embedding_providers()
        assert isinstance(providers, list)

    async def test_embedding_providers_populated(self, memory_db):
        """Test that embedding providers are properly configured."""
        providers = await memory_db.get_embedding_providers()
        assert len(providers) > 0

        # Check for default provider
        default_provider = await memory_db.get_default_embedding_provider()
        assert default_provider is not None
        assert default_provider["provider_id"] == "openai-text-embedding-3-small"
        assert default_provider["dimension"] == 1536
        assert default_provider["is_default"] is True


@pytest.mark.asyncio
class TestDocumentOperations:
    """Test document CRUD operations using high-level APIs."""

    async def test_insert_document(self, memory_manager):
        """Test inserting a document using high-level API."""
        doc = await memory_manager.add_document(
            owner_id="test_owner",
            id_at_origin="test_123",
            document_type=DocumentType.TEXT,
            document_name="Test Document",
            document_date=datetime.now(),
            metadata={"test": True},
        )

        assert doc is not None
        assert doc.owner_id == "test_owner"
        assert doc.id_at_origin == "test_123"
        assert doc.document_type == DocumentType.TEXT
        assert doc.metadata["test"] is True

    async def test_document_retrieval(self, memory_manager, memory_db):
        """Test retrieving documents using utility methods."""
        # Insert a document
        doc = await memory_manager.add_document(
            owner_id="test_owner",
            id_at_origin="test_retrieve",
            document_type=DocumentType.TEXT,
            document_name="Retrieval Test",
        )

        # Check it exists
        exists = await memory_db.document_exists(str(doc.document_id))
        assert exists is True

        # Retrieve it
        retrieved = await memory_db.get_document(str(doc.document_id))
        assert retrieved is not None
        assert retrieved["document_name"] == "Retrieval Test"

        # Count documents
        count = await memory_db.count_documents("test_owner")
        assert count >= 1

    async def test_insert_chunk_with_embedding(self, memory_manager, memory_db):
        """Test inserting a chunk with embedding vector using high-level APIs."""
        # First create a document
        doc = await memory_manager.add_document(
            owner_id="test_owner",
            id_at_origin="test_123",
            document_type=DocumentType.TEXT,
            document_name="Test Document",
        )

        # Create a test embedding (1536 dimensions for text-embedding-3-small)
        test_embedding = [0.1] * 1536

        # Insert chunk with embedding
        chunk_id = await memory_db.insert_chunk(
            document_id=str(doc.document_id),
            chunk_id=str(uuid4()),
            content="This is test content for embeddings",
            embedding=test_embedding,
            metadata={"section": "intro"},
            chunk_index=0,
            chunk_level=0,
        )

        assert chunk_id is not None

        # Verify chunk exists
        exists = await memory_db.chunk_exists(chunk_id)
        assert exists is True

        # Get chunks and verify embedding
        chunks = await memory_db.get_document_chunks(str(doc.document_id), include_embeddings=True)
        assert len(chunks) == 1
        assert chunks[0]["has_embedding"] is True

    async def test_document_deletion(self, memory_manager, memory_db):
        """Test document deletion using high-level APIs."""
        # Create a document
        doc = await memory_manager.add_document(
            owner_id="test_owner",
            id_at_origin="test_delete",
            document_type=DocumentType.TEXT,
            document_name="Delete Test",
        )

        # Add a chunk
        chunk_id = await memory_db.insert_chunk(
            document_id=str(doc.document_id),
            chunk_id=str(uuid4()),
            content="Content to be deleted",
            metadata={"test": "delete"},
        )

        # Verify both exist
        assert await memory_db.document_exists(str(doc.document_id))
        assert await memory_db.chunk_exists(chunk_id)

        # Delete document (chunks should cascade)
        deleted = await memory_db.delete_document(str(doc.document_id))
        assert deleted is True

        # Verify both are gone
        assert not await memory_db.document_exists(str(doc.document_id))
        assert not await memory_db.chunk_exists(chunk_id)


@pytest.mark.asyncio
class TestSearchOperations:
    """Test search functionality using high-level APIs."""

    async def test_text_search_with_tsvector(self, memory_manager):
        """Test full-text search using high-level search API."""
        # Process a document with searchable content
        doc, chunks = await memory_manager.process_document(
            owner_id="test_owner",
            id_at_origin="test_123",
            document_name="Test Document",
            document_type=DocumentType.TEXT,
            content="Python programming language is great for data science and machine learning",
        )

        # Search using manager's search functionality
        from llmemory.models import SearchQuery, SearchType

        query = SearchQuery(
            owner_id="test_owner",
            query_text="python programming",
            search_type=SearchType.TEXT,
            limit=10,
        )

        results = await memory_manager.search(query)

        assert len(results) > 0
        assert "Python programming" in results[0].content

    async def test_vector_search(self, memory_manager, memory_db, create_embedding):
        """Test vector search using high-level APIs."""
        # Create documents and manually add embeddings for testing
        contents = [
            "Machine learning with Python and scikit-learn",
            "Deep learning using TensorFlow and Keras",
            "Cooking recipes for Italian pasta",
        ]

        # Process documents and store their chunks
        all_chunks = []
        for content in contents:
            doc, chunks = await memory_manager.process_document(
                owner_id="test_owner",
                id_at_origin="test_vector",
                document_name=f"{content[:20]}.txt",
                document_type=DocumentType.TEXT,
                content=content,
            )
            all_chunks.extend(chunks)

        # Manually add embeddings for testing (since process_document doesn't do it synchronously)
        for chunk in all_chunks:
            embedding = await create_embedding(chunk.content)
            await memory_db.insert_chunk_embedding(str(chunk.chunk_id), embedding)

        # Now search for ML content
        query_embedding = await create_embedding("machine learning and deep learning")

        results = await memory_db.search_similar_chunks(
            owner_id="test_owner", query_embedding=query_embedding, limit=3
        )

        assert len(results) > 0
        # ML-related content should rank higher than cooking
        assert any("learning" in r["content"] for r in results[:2])

    async def test_hybrid_search(self, memory_manager, create_embedding):
        """Test hybrid search combining text and vector search."""
        # Create content
        doc, chunks = await memory_manager.process_document(
            owner_id="test_owner",
            id_at_origin="test_hybrid",
            document_name="hybrid_test.txt",
            document_type=DocumentType.TEXT,
            content="Advanced Python techniques for data analysis and visualization",
        )

        # Perform hybrid search
        query_text = "Python data"
        query_embedding = await create_embedding(query_text)

        results = await memory_manager.db.hybrid_search(
            owner_id="test_owner",
            query_text=query_text,
            query_embedding=query_embedding,
            limit=5,
            alpha=0.5,  # Equal weight to text and vector
        )

        assert len(results) > 0
        assert "Python" in results[0]["content"]

    async def test_cleanup(self, memory_db):
        """Clean up test data."""
        await memory_db.clear_all_data("test_owner")
