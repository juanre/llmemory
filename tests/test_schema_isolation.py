"""Test that llmemory correctly uses configured schema for all operations."""

from datetime import datetime
from uuid import uuid4

import pytest

from llmemory.config import DatabaseConfig as MemoryDbConfig
from llmemory.config import LLMemoryConfig
from llmemory.db import MemoryDatabase
from llmemory.library import LLMemory
from llmemory.manager import MemoryManager
from llmemory.models import DocumentType, SearchQuery, SearchType


@pytest.mark.asyncio
async def test_schema_isolation(test_db_factory):
    """Test that llmemory uses the configured schema for all database operations."""
    # Create database with custom schema using test factory
    db_manager = await test_db_factory.create_db(suffix="schema_test", schema="test_llmemory")

    # Create MemoryDatabase wrapper
    memory_db = MemoryDatabase(db_manager)
    memory_manager = MemoryManager(memory_db)

    # Initialize and apply migrations
    await memory_manager.initialize()

    # Verify tables were created in the correct schema
    async with db_manager.acquire() as conn:
        # Check that tables exist in test_llmemory schema
        tables_query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = $1
        ORDER BY table_name
        """
        tables = await conn.fetch(tables_query, "test_llmemory")
        table_names = [row["table_name"] for row in tables]

        # Should have all the llmemory tables
        expected_tables = [
            "document_chunks",
            "documents",
            "embedding_providers",
            "embedding_queue",
            "embeddings_openai_3_small",
            "search_history",
        ]
        for table in expected_tables:
            assert table in table_names, f"Table {table} not found in test_llmemory schema"

        # Check that NO tables exist in public schema (except system tables)
        public_tables = await conn.fetch(tables_query, "public")
        public_table_names = [row["table_name"] for row in public_tables]

        # None of our tables should be in public
        for table in expected_tables:
            assert (
                table not in public_table_names
            ), f"Table {table} found in public schema but should be in test_llmemory"

    # Test document operations
    owner_id = "test_owner"
    doc = await memory_manager.add_document(
        owner_id=owner_id,
        id_at_origin="test_origin",
        document_name="Test Document",
        document_type=DocumentType.TEXT,
        document_date=datetime.now(),
        metadata={"test": "metadata"},
    )

    # Verify document was created
    assert doc.document_id is not None
    assert doc.owner_id == owner_id

    # Test document processing with chunks
    doc2, chunks = await memory_manager.process_document(
        owner_id=owner_id,
        id_at_origin="test_origin_2",
        document_name="Test Document 2",
        document_type=DocumentType.TEXT,
        content="This is a test document with some content that will be chunked.",
        chunking_strategy="hierarchical",
    )

    assert len(chunks) > 0
    assert str(chunks[0].document_id) == str(doc2.document_id)

    # Test embedding operations
    test_embedding = [0.1] * 1536  # Mock embedding
    await memory_manager.update_chunk_embedding(chunks[0].chunk_id, test_embedding)

    # Test search operations
    search_query = SearchQuery(
        owner_id=owner_id, query_text="test", search_type=SearchType.VECTOR, limit=10
    )

    # Vector search
    results = await memory_manager.search(search_query, test_embedding)
    assert isinstance(results, list)

    # Test that embedding provider queries work with schema
    provider = await memory_db.get_default_embedding_provider()
    assert provider is not None
    assert provider["provider_id"] == "openai-text-embedding-3-small"

    # Test getting chunks without embeddings
    chunks_without = await memory_db.get_chunks_without_embeddings(limit=10)
    assert isinstance(chunks_without, list)

    # Verify function calls work with schema
    if chunks:
        chunk_context = await memory_manager._get_parent_context(
            chunks[0].chunk_id, context_window=2
        )
        assert isinstance(chunk_context, list)

    # Cleanup
    await memory_manager.close()


@pytest.mark.asyncio
async def test_memory_db_with_schema(test_db_factory):
    """Test MemoryDatabase operations with custom schema."""
    # Create database with custom schema
    db_manager = await test_db_factory.create_db(suffix="memory_db_test", schema="custom_schema")

    # Create and initialize MemoryDatabase
    memory_db = MemoryDatabase(db_manager)
    await memory_db.initialize()
    await memory_db.apply_migrations()

    # Test basic operations
    doc_id = str(uuid4())
    chunk_id = str(uuid4())

    # First create a document
    await memory_db.db.execute(
        """
        INSERT INTO {{tables.documents}} (document_id, owner_id, id_at_origin, document_type, document_name)
        VALUES ($1, $2, $3, $4, $5)
        """,
        doc_id,
        "test_owner",
        "test_origin",
        "text",
        "Test Document",
    )

    # Insert a chunk
    returned_id = await memory_db.insert_chunk(
        document_id=doc_id,
        chunk_id=chunk_id,
        content="Test content",
        metadata={"test": "data"},
        chunk_index=0,
        chunk_level=0,
    )

    assert str(returned_id) == chunk_id

    # Verify chunk exists
    exists = await memory_db.chunk_exists(chunk_id)
    assert exists

    # Test embedding insertion
    test_embedding = [0.1] * 1536
    success = await memory_db.insert_chunk_embedding(chunk_id=chunk_id, embedding=test_embedding)
    assert success

    # Cleanup
    await memory_db.close()


@pytest.mark.asyncio
async def test_library_with_custom_schema(test_db_factory):
    """Test LLMemory library with custom schema configuration."""
    # Create database with custom schema
    db_manager = await test_db_factory.create_db(suffix="library_schema", schema="library_test")

    # Create LLMemory instance with custom schema config
    config = LLMemoryConfig(
        database=MemoryDbConfig(schema_name="library_test", min_pool_size=5, max_pool_size=10)
    )

    memory = LLMemory(connection_string=db_manager.config.get_dsn(), config=config)

    async with memory:
        # Add a document
        result = await memory.add_document(
            owner_id="test_owner",
            id_at_origin="test_id",
            document_name="Test Doc",
            document_type=DocumentType.TEXT,
            content="This is test content for the library.",
            generate_embeddings=False,  # Skip for test
        )

        assert result.document.document_id is not None

        # Search (text-based since we didn't generate embeddings)
        results = await memory.search(
            owner_id="test_owner",
            query_text="test content",
            search_type="text",
            limit=5,
        )

        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_schema_qualified_embedding_tables(memory_db):
    """Test that embedding provider tables are correctly schema-qualified."""
    # Get the default provider
    provider = await memory_db.get_default_embedding_provider()
    assert provider is not None

    # Create a test document first
    doc_id = str(uuid4())
    chunk_id = str(uuid4())

    # Create document
    await memory_db.db.execute(
        """
        INSERT INTO {{tables.documents}} (document_id, owner_id, id_at_origin, document_type, document_name)
        VALUES ($1, $2, $3, $4, $5)
        """,
        doc_id,
        "test_owner",
        "test_origin",
        "text",
        "Test Document",
    )

    # Create a test chunk
    await memory_db.insert_chunk(
        document_id=doc_id,
        chunk_id=chunk_id,
        content="Test content for embedding",
        metadata={},
        chunk_index=0,
        chunk_level=0,
    )

    # Insert embedding
    test_embedding = [0.1] * 1536
    success = await memory_db.insert_chunk_embedding(
        chunk_id=chunk_id, embedding=test_embedding, provider_id=provider["provider_id"]
    )
    assert success

    # Test search operations that join with embedding tables
    results = await memory_db.search_similar_chunks(
        owner_id="test_owner", query_embedding=test_embedding, limit=5
    )
    assert isinstance(results, list)

    # Test hybrid search
    results = await memory_db.hybrid_search(
        owner_id="test_owner",
        query_text="test",
        query_embedding=test_embedding,
        limit=5,
    )
    assert isinstance(results, list)

    # Test getting chunks with embeddings
    chunks = await memory_db.get_document_chunks(doc_id, include_embeddings=True)
    assert isinstance(chunks, list)
    if chunks:
        assert "has_embedding" in chunks[0]
