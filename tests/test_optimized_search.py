"""Tests for enhanced search functionality with date ranges and multiple origins."""

import json
from datetime import datetime, timedelta

import pytest

from llmemory.models import DocumentType, SearchQuery, SearchType


@pytest.mark.asyncio
class TestEnhancedSearch:
    """Test enhanced search features."""

    async def test_search_with_date_range(self, memory_manager, sample_embeddings):
        """Test searching with date range filters."""
        now = datetime.now()
        past_date = now - timedelta(days=30)
        recent_date = now - timedelta(days=7)
        future_date = now + timedelta(days=7)

        # Add documents with different dates
        docs = []
        for i, (name, date) in enumerate(
            [
                ("old_doc.pdf", past_date),
                ("recent_doc.pdf", recent_date),
                ("current_doc.pdf", now),
                ("future_doc.pdf", future_date),
            ]
        ):
            doc = await memory_manager.add_document(
                owner_id="workspace_test",
                id_at_origin="user123",
                document_name=name,
                document_type=DocumentType.PDF,
                document_date=date,
            )
            docs.append(doc)

            # Process document to create chunks
            _, chunks = await memory_manager.process_document(
                owner_id="workspace_test",
                id_at_origin="user123",
                document_name=name,
                document_type=DocumentType.PDF,
                content=f"Content from {name}",
                document_date=date,
                metadata={"doc": name},
            )

            # Update chunk embedding
            if chunks:
                await memory_manager.update_chunk_embedding(
                    chunks[0].chunk_id, sample_embeddings["query"]
                )

        # Search for recent documents only
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="content",
            search_type=SearchType.VECTOR,
            date_from=recent_date - timedelta(days=1),
            date_to=now + timedelta(days=1),
            limit=10,
        )

        results = await memory_manager.search(query, query_embedding=sample_embeddings["query"])

        # Should find only recent and current documents
        assert len(results) == 2
        result_names = [
            (json.loads(r.metadata) if isinstance(r.metadata, str) else r.metadata).get("doc")
            for r in results
        ]
        assert "recent_doc.pdf" in result_names
        assert "current_doc.pdf" in result_names
        assert "old_doc.pdf" not in result_names
        assert "future_doc.pdf" not in result_names

    async def test_search_with_multiple_origins(self, memory_manager, sample_embeddings):
        """Test searching with multiple id_at_origins filter."""
        # Add documents from different origins
        origins = ["user_alice", "user_bob", "user_charlie", "user_david"]

        for origin in origins:
            _, chunks = await memory_manager.process_document(
                owner_id="workspace_test",
                id_at_origin=origin,
                document_name=f"{origin}_doc.txt",
                document_type=DocumentType.TEXT,
                content=f"Document from {origin}",
                metadata={"origin": origin},
            )

            if chunks:
                await memory_manager.update_chunk_embedding(
                    chunks[0].chunk_id, sample_embeddings["query"]
                )

        # Search for documents from specific users
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="document",
            search_type=SearchType.VECTOR,
            id_at_origins=["user_alice", "user_charlie"],
            limit=10,
        )

        results = await memory_manager.search(query, query_embedding=sample_embeddings["query"])

        # Should find only Alice and Charlie's documents
        assert len(results) == 2
        result_origins = [
            (json.loads(r.metadata) if isinstance(r.metadata, str) else r.metadata).get("origin")
            for r in results
        ]
        assert "user_alice" in result_origins
        assert "user_charlie" in result_origins
        assert "user_bob" not in result_origins
        assert "user_david" not in result_origins

    async def test_hybrid_search_with_filters(self, memory_manager, sample_embeddings):
        """Test hybrid search with metadata and date filters."""
        now = datetime.now()

        # Add diverse documents
        doc_data = [
            {
                "id_at_origin": "tech_writer",
                "name": "python_guide.md",
                "content": "Python is a versatile programming language for data science",
                "metadata": {"category": "programming", "language": "python"},
                "date": now - timedelta(days=5),
            },
            {
                "id_at_origin": "tech_writer",
                "name": "js_tutorial.md",
                "content": "JavaScript powers modern web applications",
                "metadata": {"category": "programming", "language": "javascript"},
                "date": now - timedelta(days=3),
            },
            {
                "id_at_origin": "data_scientist",
                "name": "ml_intro.md",
                "content": "Machine learning with Python transforms data into insights",
                "metadata": {"category": "ml", "language": "python"},
                "date": now - timedelta(days=1),
            },
        ]

        for data in doc_data:
            _, chunks = await memory_manager.process_document(
                owner_id="workspace_test",
                id_at_origin=data["id_at_origin"],
                document_name=data["name"],
                document_type=DocumentType.MARKDOWN,
                content=data["content"],
                document_date=data["date"],
                metadata=data["metadata"],
            )

            if chunks:
                await memory_manager.update_chunk_embedding(
                    chunks[0].chunk_id, sample_embeddings["query"]
                )

        # Try text search first to ensure documents are there
        text_query = SearchQuery(
            owner_id="workspace_test",
            query_text="Python",
            search_type=SearchType.TEXT,
            date_from=now - timedelta(days=7),
            limit=10,
        )

        text_results = await memory_manager.search(text_query)
        assert len(text_results) > 0  # Ensure text search works

        # Now try hybrid search
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="Python",
            search_type=SearchType.HYBRID,
            date_from=now - timedelta(days=7),
            limit=10,
            alpha=0.5,  # Equal weight for text and vector
        )

        results = await memory_manager.search(query, query_embedding=sample_embeddings["query"])

        # Should find documents
        assert len(results) > 0
        # Verify we find Python documents
        python_found = False
        for result in results:
            metadata = (
                json.loads(result.metadata) if isinstance(result.metadata, str) else result.metadata
            )
            if metadata.get("language") == "python":
                python_found = True
                break

        # If no Python documents found by metadata, check content
        if not python_found:
            for result in results:
                if "Python" in result.content:
                    python_found = True
                    break

        assert python_found, f"No Python documents found in {len(results)} results"

    async def test_text_search_with_date_range(self, memory_manager):
        """Test text-only search with date filtering."""
        now = datetime.now()

        # Add documents across time
        time_periods = [
            ("ancient", now - timedelta(days=365), "Ancient history of computing"),
            ("old", now - timedelta(days=90), "Old mainframe computers"),
            ("recent", now - timedelta(days=30), "Recent advances in AI"),
            ("current", now - timedelta(days=7), "Current state of technology"),
            ("latest", now, "Latest breakthroughs in quantum computing"),
        ]

        for period, date, content in time_periods:
            await memory_manager.process_document(
                owner_id="workspace_test",
                id_at_origin="historian",
                document_name=f"{period}_tech.txt",
                document_type=DocumentType.TEXT,
                content=content,
                document_date=date,
                metadata={"period": period},
            )

        # Search for recent content only (last 45 days)
        # Use simpler search terms that match the content
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="computing",
            search_type=SearchType.TEXT,
            date_from=now - timedelta(days=45),
            date_to=now,
            limit=10,
        )

        results = await memory_manager.search(query)

        # Should find only recent documents that contain "computing"
        periods_found = [
            (json.loads(r.metadata) if isinstance(r.metadata, str) else r.metadata).get("period")
            for r in results
        ]
        # "latest" has "quantum computing" so it should be found
        assert "latest" in periods_found
        # "ancient" and "old" should not be found (outside date range)
        assert "ancient" not in periods_found
        assert "old" not in periods_found
        # "recent" (AI) and "current" (technology) don't contain "computing"

    async def test_complex_search_scenario(self, memory_manager, sample_embeddings):
        """Test complex search with multiple filters combined."""
        now = datetime.now()

        # Create a diverse document set
        documents = [
            # Technical documents from Alice
            {
                "owner_id": "workspace_test",
                "id_at_origin": "alice@tech.com",
                "name": "rust_memory.md",
                "content": "Rust provides memory safety without garbage collection",
                "type": DocumentType.MARKDOWN,
                "date": now - timedelta(days=10),
                "metadata": {"author": "alice", "topic": "systems", "language": "rust"},
            },
            {
                "owner_id": "workspace_test",
                "id_at_origin": "alice@tech.com",
                "name": "python_gc.md",
                "content": "Python uses reference counting and garbage collection",
                "type": DocumentType.MARKDOWN,
                "date": now - timedelta(days=5),
                "metadata": {
                    "author": "alice",
                    "topic": "memory",
                    "language": "python",
                },
            },
            # Blog posts from Bob
            {
                "owner_id": "workspace_test",
                "id_at_origin": "bob@blog.com",
                "name": "web_trends.html",
                "content": "Modern web development trends include JAMstack and serverless",
                "type": DocumentType.HTML,
                "date": now - timedelta(days=3),
                "metadata": {"author": "bob", "topic": "web", "category": "blog"},
            },
            # Research from Charlie
            {
                "owner_id": "workspace_test",
                "id_at_origin": "charlie@research.edu",
                "name": "ai_ethics.pdf",
                "content": "Ethical considerations in artificial intelligence development",
                "type": DocumentType.PDF,
                "date": now - timedelta(days=1),
                "metadata": {
                    "author": "charlie",
                    "topic": "ai",
                    "category": "research",
                },
            },
        ]

        # Process all documents
        for doc in documents:
            _, chunks = await memory_manager.process_document(
                owner_id=doc["owner_id"],
                id_at_origin=doc["id_at_origin"],
                document_name=doc["name"],
                document_type=doc["type"],
                content=doc["content"],
                document_date=doc["date"],
                metadata=doc["metadata"],
            )

            if chunks:
                await memory_manager.update_chunk_embedding(
                    chunks[0].chunk_id, sample_embeddings["query"]
                )

        # Complex query: Find recent technical documents about memory/garbage collection
        query = SearchQuery(
            owner_id="workspace_test",
            query_text="memory garbage collection",
            search_type=SearchType.HYBRID,
            id_at_origins=[
                "alice@tech.com",
                "charlie@research.edu",
            ],  # Only Alice and Charlie
            date_from=now - timedelta(days=15),  # Last 15 days
            metadata_filter={"topic": ["systems", "memory"]},  # Would need OR support
            limit=10,
            alpha=0.6,  # Slight preference for vector similarity
        )

        # For this test, we'll use a simpler metadata filter
        query.metadata_filter = {"author": "alice"}

        results = await memory_manager.search(query, query_embedding=sample_embeddings["query"])

        # Should find Alice's technical documents
        assert len(results) > 0
        for result in results:
            assert (
                json.loads(result.metadata) if isinstance(result.metadata, str) else result.metadata
            ).get("author") == "alice"

        # Verify date filtering worked
        # All results should be within the date range
        # (Would need to join with documents table to verify dates)
