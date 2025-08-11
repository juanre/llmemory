"""Tests for enhanced chunking features."""

import asyncio

import pytest
import pytest_asyncio
from llmemory.manager import MemoryManager
from llmemory.models import DocumentType, SearchQuery, SearchType


@pytest_asyncio.fixture
async def memory_manager(memory_db):
    """Create memory manager for testing."""
    # Create a proper MemoryManager instance
    manager = MemoryManager(db=memory_db)
    await manager.initialize()
    return manager


@pytest.mark.asyncio
class TestEnhancedChunking:
    """Test enhanced chunking capabilities."""

    async def test_concurrent_document_processing(self, memory_manager):
        """Test processing multiple documents concurrently."""
        # Create test documents
        documents = [
            {
                "owner_id": "test_workspace",
                "id_at_origin": f"doc_{i}",
                "document_name": f"doc_{i}.txt",
                "document_type": DocumentType.TEXT,
                "content": f"This is document {i} about topic {i}. " * 50,
            }
            for i in range(5)
        ]

        # Process documents concurrently
        agents = []
        for doc_data in documents:
            agent = memory_manager.process_document(**doc_data)
            agents.append(agent)

        results = await asyncio.gather(*agents)

        # Verify all documents were processed
        assert len(results) == 5
        for i, (doc, chunks) in enumerate(results):
            assert doc.document_name == f"doc_{i}.txt"
            assert len(chunks) > 0

    async def test_batch_embedding_generation(self, memory_manager):
        """Test batch generation of embeddings."""
        # Add a document with multiple chunks
        doc, chunks = await memory_manager.process_document(
            owner_id="test_workspace",
            id_at_origin="batch_test",
            document_name="batch_test.txt",
            document_type=DocumentType.TEXT,
            content="This is a test document. " * 100,  # Create multiple chunks
        )

        assert len(chunks) > 1

        # Generate embeddings for all chunks (this would happen in background normally)
        # For testing, we'll just verify the chunks were created correctly
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.content is not None
            assert chunk.token_count > 0

    async def test_hierarchical_chunking(self, memory_manager):
        """Test hierarchical chunking with parent-child relationships."""
        # Create a document with clear hierarchy
        content = """
# Chapter 1: Introduction
This is the introduction to our document.

## Section 1.1: Background
Some background information here.

## Section 1.2: Overview
An overview of the topics.

# Chapter 2: Main Content
The main content starts here.

## Section 2.1: Details
Detailed information about the topic.
"""

        doc, chunks = await memory_manager.process_document(
            owner_id="test_workspace",
            id_at_origin="hierarchical_test",
            document_name="hierarchical.md",
            document_type=DocumentType.MARKDOWN,
            content=content,
        )

        # Verify chunks were created
        assert len(chunks) > 0

        # Check for hierarchical structure
        parent_chunks = [c for c in chunks if c.chunk_level == 2]
        child_chunks = [c for c in chunks if c.chunk_level == 1]

        # Should have both parent and child chunks
        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0

    async def test_search_performance(self, memory_manager):
        """Test search performance with multiple documents."""
        # Add several documents
        doc_contents = [
            "Python programming is great for data science and machine learning.",
            "JavaScript is essential for web development and browser programming.",
            "Machine learning algorithms help computers learn from data.",
            "Data science combines statistics, programming, and domain knowledge.",
            "Web development involves HTML, CSS, and JavaScript.",
        ]

        for i, content in enumerate(doc_contents):
            await memory_manager.process_document(
                owner_id="test_workspace",
                id_at_origin=f"perf_test_{i}",
                document_name=f"doc_{i}.txt",
                document_type=DocumentType.TEXT,
                content=content,
            )

        # Search for related content
        query = SearchQuery(
            owner_id="test_workspace",
            query_text="programming machine learning",
            search_type=SearchType.TEXT,
            limit=10,
        )

        results = await memory_manager.search(query)

        # Should find relevant results
        assert len(results) > 0

        # Results should be ranked by relevance
        # Documents about programming and ML should rank higher
        top_result_content = results[0].content.lower()
        assert "programming" in top_result_content or "machine learning" in top_result_content

    async def test_concurrent_search_operations(self, memory_manager):
        """Test multiple concurrent searches."""
        # First add some content
        await memory_manager.process_document(
            owner_id="test_workspace",
            id_at_origin="search_test",
            document_name="search_test.txt",
            document_type=DocumentType.TEXT,
            content="Artificial intelligence and machine learning are transforming technology.",
        )

        # Create multiple search queries
        queries = [
            SearchQuery(
                owner_id="test_workspace",
                query_text="artificial intelligence",
                search_type=SearchType.TEXT,
                limit=5,
            ),
            SearchQuery(
                owner_id="test_workspace",
                query_text="machine learning",
                search_type=SearchType.TEXT,
                limit=5,
            ),
            SearchQuery(
                owner_id="test_workspace",
                query_text="technology",
                search_type=SearchType.TEXT,
                limit=5,
            ),
        ]

        # Execute searches concurrently
        agents = [memory_manager.search(q) for q in queries]
        results = await asyncio.gather(*agents)

        # Verify all searches completed
        assert len(results) == 3
        for result_set in results:
            assert isinstance(result_set, list)
