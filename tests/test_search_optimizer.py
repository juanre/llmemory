"""Unit tests for OptimizedAsyncSearch module."""

import pytest

from llmemory.models import SearchQuery, SearchType
from llmemory.search_optimizer import OptimizedAsyncSearch


@pytest.mark.asyncio
async def test_search_optimizer_initialization(memory_manager):
    """Test OptimizedAsyncSearch initialization."""
    db = memory_manager.db.db

    optimizer = OptimizedAsyncSearch(
        db=db,
        cache_ttl=300,
        max_concurrent_queries=100,
        enable_query_optimization=True,
        hnsw_ef_search=100,
    )

    assert optimizer.cache_ttl == 300
    assert optimizer.max_concurrent_queries == 100
    assert optimizer.enable_query_optimization is True
    assert optimizer.hnsw_ef_search == 100


@pytest.mark.asyncio
async def test_vector_search_with_hnsw(memory_library_with_embeddings):
    """Test vector search uses HNSW index."""
    memory = memory_library_with_embeddings

    results = await memory.search(
        owner_id="test_workspace",
        query_text="machine learning",
        search_type=SearchType.VECTOR,
        limit=10,
    )

    assert len(results) > 0
    # All results should have similarity scores
    for result in results:
        assert result.similarity is not None
        assert 0 <= result.similarity <= 1


@pytest.mark.asyncio
async def test_text_search_with_bm25(memory_library_with_embeddings):
    """Test text search uses BM25."""
    memory = memory_library_with_embeddings

    results = await memory.search(
        owner_id="test_workspace",
        query_text="machine learning",
        search_type=SearchType.TEXT,
        limit=10,
    )

    assert len(results) > 0
    # All results should have text_rank scores
    for result in results:
        assert result.text_rank is not None


@pytest.mark.asyncio
async def test_hybrid_search_rrf_fusion(memory_library_with_embeddings):
    """Test hybrid search uses RRF fusion correctly."""
    memory = memory_library_with_embeddings

    results = await memory.search(
        owner_id="test_workspace",
        query_text="machine learning",
        search_type=SearchType.HYBRID,
        alpha=0.5,
        limit=10,
    )

    assert len(results) > 0
    # All results should have RRF scores
    for result in results:
        assert result.rrf_score is not None
        # Should also have individual scores
        assert result.similarity is not None or result.text_rank is not None
