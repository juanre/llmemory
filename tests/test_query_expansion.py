"""Tests for query expansion and multi-query search."""

import json

import pytest

from llmemory.config import LLMemoryConfig
from llmemory.library import LLMemory
from llmemory.models import DocumentType, SearchType


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
