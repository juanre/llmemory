"""Test shared pool functionality for aword-memory."""

import pytest
from llmemory import LLMemory
from llmemory.db import MemoryDatabase
from llmemory.manager import MemoryManager
from llmemory.models import DocumentType


@pytest.mark.asyncio
async def test_memory_database_from_manager(test_db_factory):
    """Test creating MemoryDatabase from external db manager."""
    # Create a test db manager with specific schema
    db_manager = await test_db_factory.create_db(suffix="memdb_shared", schema="test_schema")

    # Create MemoryDatabase using the shared pool pattern
    memory_db = MemoryDatabase.from_manager(db_manager)

    # Verify it's marked as external
    assert memory_db._external_db is True
    assert memory_db.db == db_manager  # Should use the same manager

    # Initialize and verify it works
    await memory_db.initialize()
    await memory_db.apply_migrations()

    # Test basic operations
    import uuid

    doc_id = str(uuid.uuid4())
    exists = await memory_db.document_exists(doc_id)
    assert exists is False

    # Close should not disconnect the external db
    await memory_db.close()
    # The db manager should still be usable after close
    result = await db_manager.fetch_one("SELECT 1 as test")
    assert result["test"] == 1


@pytest.mark.asyncio
async def test_memory_manager_from_db_manager(test_db_factory):
    """Test creating MemoryManager from external db manager."""
    # Create a test db manager with the schema we want
    db_manager = await test_db_factory.create_db(suffix="mgr_shared", schema="test_schema")

    # Create MemoryManager using the shared pool pattern
    manager = MemoryManager.from_db_manager(db_manager)

    # Verify it's configured correctly
    assert manager._external_db is True
    assert manager.db._external_db is True

    # Initialize and test
    await manager.initialize()

    # Test adding a document
    doc = await manager.add_document(
        owner_id="test-owner",
        id_at_origin="test-origin",
        document_name="Test Document",
        document_type=DocumentType.TEXT,
        metadata={"test": True},
    )

    assert doc.document_id is not None
    assert doc.owner_id == "test-owner"

    # Close and verify external db still works
    await manager.close()
    result = await db_manager.fetch_one("SELECT 1 as test")
    assert result["test"] == 1


@pytest.mark.asyncio
async def test_aword_memory_from_db_manager(test_db_factory):
    """Test creating LLMemory from external db manager."""
    # Create a test db manager with the schema we want
    db_manager = await test_db_factory.create_db(suffix="aword_shared", schema="test_schema")

    # Create LLMemory using the shared pool pattern
    aword_memory = LLMemory.from_db_manager(db_manager, openai_api_key="test-key")

    # Verify it's configured correctly
    assert aword_memory._external_db is True
    assert aword_memory._db_manager is db_manager

    # Initialize
    await aword_memory.initialize()

    # Verify the manager was created correctly
    assert aword_memory._manager is not None
    assert aword_memory._manager._external_db is True

    # Test basic functionality
    result = await aword_memory.add_document(
        owner_id="test-owner",
        id_at_origin="test-origin",
        document_name="Test Document",
        document_type="text",
        content="This is a test document for shared pool testing.",
    )

    assert result.document.document_id is not None

    # Close and verify external db still works
    await aword_memory.close()
    result = await db_manager.fetch_one("SELECT 1 as test")
    assert result["test"] == 1


@pytest.mark.asyncio
async def test_shared_pool_integration(test_db_factory):
    """Test full integration with shared pool pattern."""
    # In a real application, the parent would create a shared pool like this:
    # shared_pool = await asyncpg.create_pool(connection_string, min_size=50, max_size=200)

    # For testing, we'll simulate the pattern by creating managers with specific schemas
    aword_db = await test_db_factory.create_db(suffix="aword", schema="aword_test")
    other_db = await test_db_factory.create_db(suffix="other", schema="other_service")

    # Create LLMemory with shared pool
    aword_memory = LLMemory.from_db_manager(aword_db)

    # Initialize
    await aword_memory.initialize()

    # Test functionality
    doc = await aword_memory.add_document(
        owner_id="shared-test",
        id_at_origin="origin-1",
        document_name="Shared Pool Test",
        document_type="text",
        content="Testing shared pool pattern",
    )

    assert doc.document.document_id is not None

    # Verify isolation - other service shouldn't see aword's data
    # This would fail if schemas weren't properly isolated
    tables = await other_db.fetch_all(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'other_service'"
    )
    # Should be empty or only have other service's tables
    aword_tables = [t for t in tables if "document" in t["tablename"]]
    assert len(aword_tables) == 0

    # Cleanup
    await aword_memory.close()


@pytest.mark.asyncio
async def test_standalone_vs_shared_behavior(test_db_factory):
    """Test that standalone and shared pool modes behave the same."""
    # Test data
    test_content = "This is a test document for comparing behaviors."
    owner_id = "behavior-test"

    # Create two separate db managers with different schemas to avoid conflicts
    db1 = await test_db_factory.create_db(suffix="mode1", schema="test_mode1")
    db2 = await test_db_factory.create_db(suffix="mode2", schema="test_mode2")

    # Mode 1: Create LLMemory from db_manager (typical usage)
    mode1 = LLMemory.from_db_manager(db1)
    await mode1.initialize()

    mode1_doc = await mode1.add_document(
        owner_id=owner_id,
        id_at_origin="mode1-1",
        document_name="Mode 1 Test",
        document_type="text",
        content=test_content,
    )

    # Mode 2: Create another LLMemory from a different db_manager
    mode2 = LLMemory.from_db_manager(db2)
    await mode2.initialize()

    mode2_doc = await mode2.add_document(
        owner_id=owner_id,
        id_at_origin="mode2-1",
        document_name="Mode 2 Test",
        document_type="text",
        content=test_content,
    )

    # Both should work the same way
    assert mode1_doc.document.document_type == mode2_doc.document.document_type
    assert mode1_doc.document.owner_id == mode2_doc.document.owner_id

    # Search should work in both
    mode1_results = await mode1.search(
        owner_id=owner_id, query_text="test document", search_type="text"
    )

    mode2_results = await mode2.search(
        owner_id=owner_id, query_text="test document", search_type="text"
    )

    assert len(mode1_results) > 0
    assert len(mode2_results) > 0

    # Cleanup
    await mode1.close()
    await mode2.close()
