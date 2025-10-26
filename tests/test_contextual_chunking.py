# ABOUTME: Tests that contextual retrieval metadata is set during chunking, not embedding.
# ABOUTME: Verifies chunks get contextualized flag even without embeddings, ensuring correct architecture.

import pytest

from llmemory import DocumentType, LLMemory, LLMemoryConfig


@pytest.mark.asyncio
async def test_contextual_metadata_set_during_chunking(test_db_factory):
    """Test that contextualized metadata is set during chunking, not embedding."""

    config = LLMemoryConfig()
    config.chunking.enable_contextual_retrieval = True

    db_manager = await test_db_factory.create_db(suffix="chunk_meta", schema="llmemory")
    memory = LLMemory(connection_string=db_manager.config.get_dsn(), config=config)
    await memory.initialize()

    # Add document WITHOUT generating embeddings
    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Test Document",
        document_type=DocumentType.REPORT,
        content="This is test content for chunking.",
        generate_embeddings=False,  # Key: no embeddings
    )

    # Get chunks - they should still have contextualized flag
    chunks = await memory.get_document_chunks(result.document.document_id)

    # Assert metadata flag was set during chunking, not embedding
    assert len(chunks) > 0, "Should have chunks"
    assert (
        chunks[0].metadata.get("contextualized") is True
    ), "Chunks should be marked as contextualized even without embeddings"

    await memory.close()


@pytest.mark.asyncio
async def test_contextual_metadata_not_set_when_disabled(test_db_factory):
    """Test that contextualized metadata is NOT set when feature is disabled."""

    config = LLMemoryConfig()
    config.chunking.enable_contextual_retrieval = False  # Disabled

    db_manager = await test_db_factory.create_db(suffix="no_context", schema="llmemory")
    memory = LLMemory(connection_string=db_manager.config.get_dsn(), config=config)
    await memory.initialize()

    # Add document
    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Test Document",
        document_type=DocumentType.REPORT,
        content="This is test content.",
        generate_embeddings=False,
    )

    # Get chunks - they should NOT have contextualized flag
    chunks = await memory.get_document_chunks(result.document.document_id)

    assert len(chunks) > 0, "Should have chunks"
    assert (
        chunks[0].metadata.get("contextualized") is not True
    ), "Chunks should not be marked as contextualized when feature is disabled"

    await memory.close()
