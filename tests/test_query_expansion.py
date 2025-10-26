"""Tests for query expansion and multi-query search."""

import json

import pytest
from unittest.mock import AsyncMock, Mock

from llmemory.config import LLMemoryConfig, SearchConfig
from llmemory.library import LLMemory
from llmemory.models import DocumentType, SearchType
from llmemory.query_expansion import QueryExpansionService


@pytest.mark.asyncio
async def test_multi_query_search_logging(memory_db):
    """Ensure multi-query search generates variants and logs diagnostics."""
    config = LLMemoryConfig()
    config.search.enable_query_expansion = True
    config.search.max_query_variants = 3
    config.search.include_keyword_variant = True

    memory = LLMemory(
        connection_string=memory_db.db.config.get_dsn(),
        config=config,
    )
    await memory.initialize()

    try:
        # Ingest sample documents
        owner_id = "workspace_test"
        id_origin = "doc"

        contents = [
            (
                "python_data_analysis.txt",
                "Python data analysis involves pandas, NumPy, and visualization to understand datasets.",
            ),
            (
                "machine_learning.txt",
                "Machine learning techniques include supervised learning, regression, and classification.",
            ),
        ]

        for idx, (name, content) in enumerate(contents, start=1):
            await memory.add_document(
                owner_id=owner_id,
                id_at_origin=f"{id_origin}_{idx}",
                document_name=name,
                document_type=DocumentType.TEXT,
                content=content,
                generate_embeddings=False,
            )

        results = await memory.search(
            owner_id=owner_id,
            query_text="How to learn data analysis in Python?",
            search_type=SearchType.TEXT,
            limit=5,
            query_expansion=True,
            rerank=True,
        )

        assert results, "Expected search results with multi-query expansion"

        query = memory._manager.db.db_manager.prepare_query(
            """
            SELECT results
            FROM {{tables.search_history}}
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = await memory._manager.db.db_manager.fetch_one(query)
        assert row is not None

        payload = json.loads(row["results"])
        diagnostics = payload.get("diagnostics", {})

        variants = diagnostics.get("query_variants", [])
        assert len(variants) >= 2, "Expected multiple query variants in diagnostics"
        assert diagnostics.get("backend") == "multi_query"
        assert diagnostics.get("rerank_applied") is True

    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_single_query_rerank_logging(memory_db):
    """Ensure single-query rerank logs diagnostics."""
    config = LLMemoryConfig()
    config.search.enable_query_expansion = False
    config.search.enable_rerank = True
    config.search.rerank_top_k = 5
    config.search.rerank_return_k = 3

    memory = LLMemory(
        connection_string=memory_db.db.config.get_dsn(),
        config=config,
    )
    await memory.initialize()

    try:
        owner_id = "workspace_test"

        await memory.add_document(
            owner_id=owner_id,
            id_at_origin="doc1",
            document_name="lexical.txt",
            document_type=DocumentType.TEXT,
            content="Python data pipelines rely on pandas and airflow.",
            generate_embeddings=False,
        )

        results = await memory.search(
            owner_id=owner_id,
            query_text="python data pipelines",
            search_type=SearchType.TEXT,
            limit=5,
            rerank=True,
        )

        assert results, "Expected reranked search results"

        query = memory._manager.db.db_manager.prepare_query(
            """
            SELECT results
            FROM {{tables.search_history}}
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = await memory._manager.db.db_manager.fetch_one(query)
        assert row is not None

        payload = json.loads(row["results"])
        diagnostics = payload.get("diagnostics", {})

        assert diagnostics.get("backend") == "text"
        assert diagnostics.get("rerank_requested") is True
        assert diagnostics.get("rerank_applied") is True
        assert diagnostics.get("rerank_top_k") == config.search.rerank_top_k

    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_chunk_summary_generation(memory_db):
    """Summaries should be generated and returned with search results when enabled."""
    config = LLMemoryConfig()
    config.chunking.enable_chunk_summaries = True
    config.search.enable_query_expansion = False

    memory = LLMemory(
        connection_string=memory_db.db.config.get_dsn(),
        config=config,
    )
    await memory.initialize()

    try:
        owner_id = "workspace_test"

        await memory.add_document(
            owner_id=owner_id,
            id_at_origin="doc1",
            document_name="long_text.txt",
            document_type=DocumentType.TEXT,
            content=(
                "Python enables rapid experimentation. "
                "Its extensive ecosystem of libraries makes data analysis accessible. "
                "Teams rely on readable syntax when collaborating on complex projects."
            ),
            generate_embeddings=False,
        )

        results = await memory.search(
            owner_id=owner_id,
            query_text="Python experimentation",
            search_type=SearchType.TEXT,
            limit=3,
        )

        assert results, "Expected search results with summaries"
        top = results[0]
        assert top.summary is not None and len(top.summary) > 0
        assert top.metadata.get("summary") == top.summary

    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_llm_query_expansion_callback_is_invoked(memory_db):
    """Verify that LLM callback is called when query_expansion=True."""
    config = LLMemoryConfig()
    config.search.enable_query_expansion = True

    memory = LLMemory(
        connection_string=memory_db.db.config.get_dsn(),
        config=config,
    )
    await memory.initialize()

    try:
        owner_id = "test-owner"

        # Add a test document
        await memory.add_document(
            owner_id=owner_id,
            id_at_origin="test-doc",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
            content="Machine learning is a subset of artificial intelligence.",
            generate_embeddings=False,
        )

        # Track if callback was invoked
        call_count = 0

        async def mock_llm_callback(query: str, limit: int):
            nonlocal call_count
            call_count += 1
            return [
                "semantic variant one",
                "semantic variant two"
            ]

        # Wire the callback (this will fail - method doesn't exist yet)
        memory._query_expander.llm_callback = mock_llm_callback

        results = await memory.search(
            owner_id=owner_id,
            query_text="test query",
            search_type=SearchType.TEXT,
            query_expansion=True,
            max_query_variants=3,
            limit=5
        )

        # Verify callback was invoked
        assert call_count > 0, "LLM callback should be invoked"

    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_llm_expansion_auto_wired_from_config(memory_db):
    """Verify LLMemory automatically wires LLM callback when configured."""
    from llmemory.query_expansion import QueryExpansionService

    # Configure LLM expansion model
    config = LLMemoryConfig()
    config.search.enable_query_expansion = True
    config.search.query_expansion_model = "gpt-4o-mini"

    memory = LLMemory(
        connection_string=memory_db.db.config.get_dsn(),
        openai_api_key="sk-test-key",
        config=config
    )
    await memory.initialize()

    try:
        # Verify query expander has LLM callback wired
        assert memory._query_expander is not None
        assert memory._query_expander.llm_callback is not None, \
            "LLM callback should be auto-wired when query_expansion_model configured"

    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_query_expansion_heuristic_variants():
    """Test heuristic variant generation."""
    config = SearchConfig()
    config.max_query_variants = 3
    config.include_keyword_variant = True

    service = QueryExpansionService(config)

    variants = await service.expand("how to improve customer satisfaction", max_variants=3)

    # Should get 3 variants
    assert len(variants) <= 3

    # Should include OR variant
    assert any("OR" in v for v in variants), "Should include OR variant"

    # Should include quoted variant
    assert any(v.startswith('"') and v.endswith('"') for v in variants), \
        "Should include quoted variant"

    # Variants should not include original
    assert "how to improve customer satisfaction" not in [v.lower() for v in variants]


@pytest.mark.asyncio
async def test_query_expansion_with_llm_callback():
    """Test LLM callback is preferred over heuristics."""
    config = SearchConfig()
    config.max_query_variants = 3

    # Create mock callback
    async def mock_llm(query: str, limit: int):
        return [
            "semantic variant one",
            "semantic variant two",
            "semantic variant three"
        ]

    service = QueryExpansionService(config, llm_callback=mock_llm)

    variants = await service.expand("test query", max_variants=3)

    # Should use LLM variants, not heuristics
    assert "semantic variant one" in variants
    assert "semantic variant two" in variants
    assert len(variants) == 3


@pytest.mark.asyncio
async def test_query_expansion_fallback_on_llm_failure():
    """Test fallback to heuristics when LLM callback fails."""
    config = SearchConfig()
    config.max_query_variants = 3
    config.include_keyword_variant = True

    # Create failing callback
    async def failing_llm(query: str, limit: int):
        raise Exception("LLM API failure")

    service = QueryExpansionService(config, llm_callback=failing_llm)

    variants = await service.expand("test query expansion", max_variants=3)

    # Should fall back to heuristics
    assert len(variants) > 0, "Should fall back to heuristics on LLM failure"
    assert any("OR" in v for v in variants), "Should include heuristic OR variant"


@pytest.mark.asyncio
async def test_query_expansion_timeout():
    """Test LLM callback timeout (8 seconds)."""
    import asyncio

    config = SearchConfig()

    # Create slow callback
    async def slow_llm(query: str, limit: int):
        await asyncio.sleep(10)  # Exceeds 8 second timeout
        return ["variant"]

    service = QueryExpansionService(config, llm_callback=slow_llm)

    # Should timeout and fall back to heuristics
    variants = await service.expand("test query", max_variants=2)

    # Should have heuristic variants (fallback)
    assert len(variants) > 0
