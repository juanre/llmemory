# ABOUTME: Comprehensive error handling tests for llmemory validating all exception types and error paths.
# ABOUTME: Tests ValidationError, DocumentNotFoundError, EmbeddingError, and ConfigurationError scenarios.

"""Comprehensive error handling tests for llmemory."""

from uuid import UUID

import pytest

from llmemory import (
    ConfigurationError,
    DatabaseError,
    DocumentNotFoundError,
    DocumentType,
    EmbeddingError,
    LLMemory,
    SearchError,
    ValidationError,
)
from llmemory.config import LLMemoryConfig


@pytest.mark.asyncio
async def test_validation_error_on_empty_owner_id(test_db_factory):
    """Test ValidationError raised for empty owner_id."""
    db_manager = await test_db_factory.create_db(suffix="error_test1", schema="llmemory")
    memory = LLMemory(connection_string=db_manager.config.get_dsn())
    await memory.initialize()

    with pytest.raises(ValidationError) as exc_info:
        await memory.add_document(
            owner_id="",  # Invalid
            id_at_origin="test",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
            content="test content that is long enough to pass validation",
        )

    assert "owner_id" in str(exc_info.value).lower()

    await memory.close()


@pytest.mark.asyncio
async def test_validation_error_on_invalid_owner_id_pattern(test_db_factory):
    """Test ValidationError for invalid owner_id characters."""
    db_manager = await test_db_factory.create_db(suffix="error_test2", schema="llmemory")
    memory = LLMemory(connection_string=db_manager.config.get_dsn())
    await memory.initialize()

    with pytest.raises(ValidationError):
        await memory.add_document(
            owner_id="workspace@invalid!chars",  # Invalid pattern
            id_at_origin="test",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
            content="test content that is long enough to pass validation",
        )

    await memory.close()


@pytest.mark.asyncio
async def test_document_not_found_error(test_db_factory):
    """Test DocumentNotFoundError for non-existent documents."""
    db_manager = await test_db_factory.create_db(suffix="error_test3", schema="llmemory")
    memory = LLMemory(connection_string=db_manager.config.get_dsn())
    await memory.initialize()

    with pytest.raises(DocumentNotFoundError):
        await memory.get_document("test", UUID("00000000-0000-0000-0000-000000000000"))

    await memory.close()


@pytest.mark.asyncio
async def test_embedding_error_on_invalid_api_key(test_db_factory):
    """Test EmbeddingError when OpenAI API key is invalid.

    The system is resilient - documents are added even if embeddings fail.
    We verify the embedding failed but document was created.
    """
    db_manager = await test_db_factory.create_db(suffix="error_test4", schema="llmemory")
    memory = LLMemory(connection_string=db_manager.config.get_dsn(), openai_api_key="sk-invalid")
    await memory.initialize()

    # Document should be added successfully even if embeddings fail
    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="test content that is long enough to pass validation",
        generate_embeddings=True,
    )

    # Document was created successfully
    assert result is not None
    assert result.document.document_name == "test.txt"
    # But no embeddings were generated due to invalid API key
    assert result.embeddings_created == 0
    # Chunks were still created
    assert result.chunks_created > 0

    await memory.close()


@pytest.mark.asyncio
async def test_configuration_error_on_missing_connection():
    """Test ConfigurationError when no connection info provided."""
    with pytest.raises(ConfigurationError):
        memory = LLMemory()  # No connection_string or db_manager
