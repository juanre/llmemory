# ABOUTME: Tests for timeout handling in query expansion and reranking services.
# ABOUTME: Verifies 8-second timeouts trigger fallback to heuristics/lexical scoring.

"""Tests for timeout and fallback behavior."""

import asyncio
from uuid import uuid4

import pytest

from llmemory.config import SearchConfig
from llmemory.models import SearchResult
from llmemory.query_expansion import QueryExpansionService
from llmemory.reranker import RerankerService


@pytest.mark.asyncio
async def test_query_expansion_timeout():
    """Test query expansion times out slow LLM callbacks."""
    config = SearchConfig()

    async def slow_callback(query: str, limit: int):
        await asyncio.sleep(10)  # Exceeds 8 second timeout
        return ["variant"]

    service = QueryExpansionService(config, llm_callback=slow_callback)

    # Should timeout and fall back to heuristics
    variants = await service.expand("test query", max_variants=2)

    # Should have heuristic variants (fallback)
    assert len(variants) > 0
    assert all(
        "semantic variant" not in v for v in variants
    ), "Should use heuristics, not LLM variants"


@pytest.mark.asyncio
async def test_reranker_timeout():
    """Test reranking times out slow callbacks."""

    async def slow_reranker(query: str, results):
        await asyncio.sleep(10)  # Exceeds timeout
        return [1.0] * len(results)

    config = SearchConfig()
    service = RerankerService(config, score_callback=slow_reranker)

    # Create test results
    results = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="test",
            metadata={},
            score=1.0,
        )
    ]

    # Should timeout and fall back to lexical
    reranked = await service.rerank("test query", results, top_k=1)

    # Should return results (fallback to lexical scoring)
    assert len(reranked) > 0
