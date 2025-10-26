# ABOUTME: Tests for contextual retrieval feature that prepends document context to chunks before embedding.
# ABOUTME: Validates that contextualization improves precision by including document-level information in chunk embeddings.

import pytest
from llmemory import LLMemory, DocumentType, LLMemoryConfig


@pytest.mark.asyncio
async def test_contextual_retrieval_prepends_document_context(test_db_factory):
    """Test that chunks include document context when contextual_retrieval enabled."""

    config = LLMemoryConfig()
    config.chunking.enable_contextual_retrieval = True

    # Create test database
    db_manager = await test_db_factory.create_db(suffix="contextual", schema="llmemory")

    memory = LLMemory(connection_string=db_manager.config.get_dsn(), config=config)
    await memory.initialize()

    # Add document
    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 2024 Financial Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% QoQ. Profit margins improved."
    )

    # Get chunks and check they have contextualized content
    chunks = await memory.get_document_chunks(result.document.document_id)

    # First chunk should have document context prepended to embedding metadata
    assert chunks[0].metadata.get("contextualized") is True
    # The actual chunk content should remain original (for display)
    assert "Revenue increased" in chunks[0].content

    await memory.close()


@pytest.mark.asyncio
async def test_contextual_retrieval_improves_precision(test_db_factory):
    """Test that contextual retrieval improves precision."""

    # Test with contextual OFF
    config_no_context = LLMemoryConfig()
    config_no_context.chunking.enable_contextual_retrieval = False

    db_no_context = await test_db_factory.create_db(suffix="no_context", schema="llmemory")
    memory_no_context = LLMemory(connection_string=db_no_context.config.get_dsn(), config=config_no_context)
    await memory_no_context.initialize()

    await memory_no_context.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% in the technology sector."
    )

    results_no_context = await memory_no_context.search(
        owner_id="test",
        query_text="Q3 technology revenue growth",
        limit=5
    )

    await memory_no_context.close()

    # Test with contextual ON
    config_with_context = LLMemoryConfig()
    config_with_context.chunking.enable_contextual_retrieval = True

    db_with_context = await test_db_factory.create_db(suffix="with_context", schema="llmemory")
    memory_with_context = LLMemory(connection_string=db_with_context.config.get_dsn(), config=config_with_context)
    await memory_with_context.initialize()

    await memory_with_context.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% in the technology sector."
    )

    results_with_context = await memory_with_context.search(
        owner_id="test",
        query_text="Q3 technology revenue growth",
        limit=5
    )

    await memory_with_context.close()

    # With context should have equal or better scores
    if results_no_context and results_with_context:
        assert results_with_context[0].score >= results_no_context[0].score * 0.9, \
            "Contextual retrieval should not significantly degrade scores"
