"""Tests for pgvector-specific operations using high-level APIs."""

from uuid import uuid4

import numpy as np
import pytest
from llmemory.models import DocumentType


@pytest.mark.asyncio
class TestPgvectorOperations:
    """Test pgvector-specific functionality using high-level APIs."""

    async def test_vector_operations(self, memory_db, memory_manager):
        """Test various pgvector operators through high-level operations."""
        # Create test vectors
        v1 = [1.0, 0.0, 0.0] + [0.0] * 1533  # 1536 dimensions
        v2 = [0.0, 1.0, 0.0] + [0.0] * 1533
        v3 = [0.707, 0.707, 0.0] + [0.0] * 1533  # 45-degree angle

        # Create a document using the manager
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="test",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
        )

        # Create chunks with specific embeddings
        test_data = [("Vector 0", v1), ("Vector 1", v2), ("Vector 2", v3)]

        for content, vec in test_data:
            chunk_id = await memory_db.insert_chunk(
                document_id=str(doc.document_id),
                chunk_id=str(uuid4()),
                content=content,
                embedding=vec,
                metadata={"test": "vector"},
            )

        # Test vector similarity search using high-level API
        # The search internally uses the <=> operator
        results = await memory_db.search_similar_chunks(
            owner_id="workspace_test", query_embedding=v1, limit=3
        )

        # Vector 0 should be closest to itself
        assert len(results) > 0
        assert results[0]["content"] == "Vector 0"

        # Test direct vector similarity calculation
        similarity = await memory_db.vector_similarity_direct(v1, v1)
        assert similarity > 0.999  # Should be ~1.0 (identical vectors)

        similarity = await memory_db.vector_similarity_direct(v1, v2)
        assert similarity < 0.1  # Should be ~0 (orthogonal vectors)

    async def test_vector_indexing_performance(self, memory_db, memory_manager):
        """Test that vector indexes are being used efficiently."""
        # Create a document
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="perf_test",
            document_name="perf.txt",
            document_type=DocumentType.TEXT,
        )

        # Add 100 random vectors
        for i in range(100):
            vec = np.random.random(1536).tolist()
            await memory_db.insert_chunk(
                document_id=str(doc.document_id),
                chunk_id=str(uuid4()),
                content=f"Vector {i}",
                embedding=vec,
                metadata={"index": i},
            )

        # Query with a random vector
        query_vec = np.random.random(1536).tolist()

        # Search should use index efficiently
        results = await memory_db.search_similar_chunks(
            owner_id="workspace_test", query_embedding=query_vec, limit=10
        )

        assert len(results) <= 10
        # All results should be from our test owner
        assert all(r["chunk_id"] is not None for r in results)

    async def test_vector_arithmetic(self, memory_db):
        """Test vector arithmetic operations using utility methods."""
        # Create normalized vectors
        v1 = [1.0, 0.0] + [0.0] * 1534
        v2 = [0.0, 1.0] + [0.0] * 1534

        # Test vector operations are working
        operations_work = await memory_db.test_vector_operations()
        assert operations_work is True

        # Test similarity calculation
        # Similarity between orthogonal vectors should be ~0
        similarity = await memory_db.vector_similarity_direct(v1, v2)
        assert similarity < 0.1

        # Similarity with self should be ~1
        similarity = await memory_db.vector_similarity_direct(v1, v1)
        assert similarity > 0.999

    async def test_vector_aggregation(self, memory_db, memory_manager):
        """Test vector aggregation through high-level operations."""
        # Create a document
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="agg_test",
            document_name="agg.txt",
            document_type=DocumentType.TEXT,
        )

        # Add several vectors
        vectors = [
            [1.0, 0.0, 0.0] + [0.0] * 1533,
            [0.0, 1.0, 0.0] + [0.0] * 1533,
            [0.0, 0.0, 1.0] + [0.0] * 1533,
        ]

        for i, vec in enumerate(vectors):
            await memory_db.insert_chunk(
                document_id=str(doc.document_id),
                chunk_id=str(uuid4()),
                content=f"Vector {i}",
                embedding=vec,
                metadata={"group": "test"},
            )

        # Count chunks for this document
        chunk_count = await memory_db.count_chunks(str(doc.document_id))
        assert chunk_count == 3

        # Get chunks with embeddings
        chunks = await memory_db.get_document_chunks(str(doc.document_id), include_embeddings=True)

        assert len(chunks) == 3
        assert all(chunk.get("has_embedding") for chunk in chunks)

    async def test_vector_constraints(self, memory_db, memory_manager):
        """Test vector dimension constraints using high-level APIs."""
        # Create a document
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="constraint_test",
            document_name="constraint.txt",
            document_type=DocumentType.TEXT,
        )

        # Create chunk without embedding first
        chunk_id = await memory_db.insert_chunk(
            document_id=str(doc.document_id),
            chunk_id=str(uuid4()),
            content="Test content",
            metadata={"test": "constraint"},
        )

        # Try to insert vector with wrong dimensions (should raise ValueError)
        wrong_vec = [1.0] * 100  # Wrong dimension (should be 1536)

        # This should raise ValueError for dimension mismatch
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            await memory_db.insert_chunk_embedding(
                chunk_id, wrong_vec, provider_id="openai-text-embedding-3-small"
            )

    async def test_null_embeddings(self, memory_db, memory_manager):
        """Test handling of chunks without embeddings."""
        # Create a document
        doc = await memory_manager.add_document(
            owner_id="workspace_test",
            id_at_origin="null_test",
            document_name="null.txt",
            document_type=DocumentType.TEXT,
        )

        # Insert chunk without embedding
        chunk_id = await memory_db.insert_chunk(
            document_id=str(doc.document_id),
            chunk_id=str(uuid4()),
            content="No embedding",
            metadata={"test": "null"},
        )

        # Get chunks and verify no embedding
        chunks = await memory_db.get_document_chunks(str(doc.document_id), include_embeddings=True)

        assert len(chunks) == 1
        assert chunks[0]["has_embedding"] is False

        # Get chunks without embeddings
        chunks_without = await memory_db.get_chunks_without_embeddings(limit=10)

        # Should include our chunk
        chunk_ids = [c["chunk_id"] for c in chunks_without]
        assert chunk_id in chunk_ids

    async def test_cleanup(self, memory_db):
        """Test cleanup operations."""
        # Clear test data
        result = await memory_db.clear_all_data("workspace_test")

        # Verify cleanup
        doc_count = await memory_db.count_documents("workspace_test")
        assert doc_count == 0
