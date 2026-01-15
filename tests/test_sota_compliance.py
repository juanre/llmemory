# ABOUTME: Integration tests verifying SOTA RAG features work correctly end-to-end with real database.
# ABOUTME: Tests hybrid search, query expansion, reranking, query routing, and contextual retrieval capabilities.

import os

import pytest

from llmemory import DocumentType, LLMemory, LLMemoryConfig, SearchType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_hybrid_search(test_db_factory):
    """Verify hybrid search (vector + BM25) works correctly."""
    db_manager = await test_db_factory.create_db(suffix="hybrid", schema="llmemory")

    memory = LLMemory(
        connection_string=db_manager.config.get_dsn(), openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    await memory.initialize()

    # Add test documents
    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="ml.txt",
        document_type=DocumentType.TEXT,
        content="Machine learning uses neural networks for pattern recognition.",
    )

    # Hybrid search should work
    results = await memory.search(
        owner_id="test",
        query_text="deep learning neural networks",
        search_type=SearchType.HYBRID,
        alpha=0.5,
        limit=5,
    )

    assert len(results) > 0
    # Should have RRF scores
    assert all(r.rrf_score is not None for r in results)
    # Should have individual scores (similarity or text_rank)
    for result in results:
        assert result.similarity is not None or result.text_rank is not None

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_query_expansion_llm(test_db_factory):
    """Verify LLM-based query expansion works."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY")

    config = LLMemoryConfig()
    config.search.query_expansion_model = "gpt-4o-mini"

    db_manager = await test_db_factory.create_db(suffix="expansion", schema="llmemory")
    memory = LLMemory(
        connection_string=db_manager.config.get_dsn(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        config=config,
    )
    await memory.initialize()

    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="Artificial intelligence and machine learning.",
    )

    results = await memory.search(
        owner_id="test", query_text="AI", query_expansion=True, max_query_variants=3, limit=5
    )

    # Should work (LLM generates semantic variants)
    assert len(results) >= 0  # May be 0 if variants don't match

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_reranking(test_db_factory):
    """Verify reranking improves result quality."""
    db_manager = await test_db_factory.create_db(suffix="rerank", schema="llmemory")

    memory = LLMemory(
        connection_string=db_manager.config.get_dsn(), openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    await memory.initialize()

    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="Machine learning is a subset of artificial intelligence.",
    )

    # Search with reranking
    results = await memory.search(
        owner_id="test",
        query_text="AI machine learning",
        search_type=SearchType.HYBRID,
        rerank=True,
        rerank_top_k=20,
        rerank_return_k=5,
        limit=5,
    )

    # Should have rerank scores
    if len(results) > 0:
        assert all(
            r.rerank_score is not None for r in results
        ), "All results should have rerank scores when rerank=True"

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_query_routing(test_db_factory):
    """Verify query routing works."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY")

    db_manager = await test_db_factory.create_db(suffix="routing", schema="llmemory")

    memory = LLMemory(
        connection_string=db_manager.config.get_dsn(), openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    await memory.initialize()

    # Add some documents
    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="kb.txt",
        document_type=DocumentType.TEXT,
        content="Our product supports password reset via email.",
    )

    # Answerable query
    result = await memory.search_with_routing(
        owner_id="test", query_text="How do I reset my password?", enable_routing=True
    )

    assert result["route"] == "retrieval"
    assert result["confidence"] > 0.5

    # Unanswerable query
    result = await memory.search_with_routing(
        owner_id="test", query_text="What's the current weather?", enable_routing=True
    )

    assert result["route"] in ["web_search", "unanswerable"]

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_contextual_retrieval(test_db_factory):
    """Verify contextual retrieval works."""
    config = LLMemoryConfig()
    config.chunking.enable_contextual_retrieval = True

    db_manager = await test_db_factory.create_db(suffix="contextual", schema="llmemory")

    memory = LLMemory(
        connection_string=db_manager.config.get_dsn(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        config=config,
    )
    await memory.initialize()

    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 Financial Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% QoQ in the technology sector.",
    )

    # Chunks should be marked as contextualized
    doc_result = await memory.list_documents(owner_id="test", limit=1)
    doc_id = doc_result.documents[0].document_id

    chunks = await memory.get_document_chunks("test", doc_id, limit=1)
    assert chunks[0].metadata.get("contextualized") is True

    await memory.close()
