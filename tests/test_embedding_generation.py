"""Tests for OpenAI integration."""

import pytest

from llmemory.models import DocumentType, SearchType


@pytest.mark.asyncio
class TestOpenAIIntegration:
    """Test real OpenAI embedding functionality."""

    async def test_openai_embeddings_generation(self, openai_client):
        """Test that we can generate embeddings using OpenAI."""
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input="Test text for embedding generation"
        )

        embedding = response.data[0].embedding

        # Verify embedding properties
        assert len(embedding) == 1536  # text-embedding-3-small dimension
        assert all(isinstance(x, float) for x in embedding)

        # Verify embedding is normalized (approximately)
        norm = sum(x**2 for x in embedding) ** 0.5
        assert 0.99 < norm < 1.01  # Should be close to 1.0

    async def test_embedding_similarity(self, sample_embeddings):
        """Test that similar texts have similar embeddings."""
        import numpy as np

        # Calculate cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Similar texts should have high similarity
        sim_query_similar = cosine_similarity(
            sample_embeddings["query"], sample_embeddings["similar"]
        )

        # Different texts should have lower similarity
        sim_query_different = cosine_similarity(
            sample_embeddings["query"], sample_embeddings["different"]
        )

        # Verify relationships
        assert sim_query_similar > sim_query_different
        assert sim_query_similar > 0.7  # Should be reasonably similar
        assert sim_query_different < 0.5  # Should be quite different

    async def test_search_with_real_embeddings(self, memory_library_with_embeddings):
        """Test search functionality with real embeddings."""
        memory = memory_library_with_embeddings

        # The fixture has already added documents with embeddings
        # Now search for them
        results = await memory.search(
            owner_id="test_workspace",
            query_text="artificial intelligence and machine learning",
            search_type=SearchType.VECTOR,
            limit=5,
        )

        assert len(results) > 0
        assert results[0].similarity is not None
        assert results[0].similarity > 0.5  # Should have reasonable similarity

        # The most relevant document should be ranked first
        assert "artificial intelligence" in results[0].content.lower()

    async def test_hybrid_search_with_real_embeddings(self, memory_library_with_embeddings):
        """Test hybrid search combining vector and text search."""
        memory = memory_library_with_embeddings

        # Perform hybrid search
        results = await memory.search(
            owner_id="test_workspace",
            query_text="Python programming",
            search_type=SearchType.HYBRID,
            limit=5,
            alpha=0.5,  # Equal weight to vector and text
        )

        assert len(results) > 0
        assert results[0].rrf_score is not None

        # Should find Python-related content
        found_python = any("python" in r.content.lower() for r in results)
        assert found_python

    async def test_batch_embedding_generation(self, memory_library):
        """Test generating embeddings for multiple texts at once."""
        # Add multiple documents
        docs = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
            "Natural language processing helps computers understand human language",
        ]

        # Add documents
        doc_ids = []
        for i, content in enumerate(docs):
            result = await memory_library.add_document(
                owner_id="test_workspace",
                id_at_origin=f"batch_test_{i}",
                document_name=f"batch_{i}.txt",
                document_type=DocumentType.TEXT,
                content=content,
                generate_embeddings=True,  # Generate embeddings immediately
            )
            doc_ids.append(result.document.document_id)

        # Give a moment for embeddings to be generated
        import asyncio

        await asyncio.sleep(1)

        # Search to verify embeddings were generated
        results = await memory_library.search(
            owner_id="test_workspace",
            query_text="machine learning artificial intelligence",
            search_type=SearchType.VECTOR,
            limit=10,
        )

        # Should find our documents
        assert len(results) > 0
        result_doc_ids = [r.document_id for r in results]

        # At least one of our documents should be in results
        assert any(doc_id in result_doc_ids for doc_id in doc_ids)
