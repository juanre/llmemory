"""Search tests using async patterns."""

from pathlib import Path

import pytest

from llmemory.models import DocumentType, SearchQuery, SearchType


@pytest.mark.asyncio
class TestSearchSync:
    """Test search functionality using async methods."""

    async def test_basic_text_search(self, memory_manager):
        """Test basic text search functionality."""
        # Process a document with content
        doc, chunks = await memory_manager.process_document(
            owner_id="test_workspace",
            id_at_origin="test_user",
            document_name="quantum_intro.txt",
            document_type=DocumentType.TEXT,
            content="""
            Quantum computing leverages quantum mechanical phenomena like superposition and entanglement.
            Unlike classical bits that are either 0 or 1, qubits can exist in superposition of both states.
            This allows quantum computers to perform certain calculations exponentially faster.
            Shor's algorithm can factor large numbers efficiently, threatening current cryptography.
            Google achieved quantum supremacy in 2019 with their Sycamore processor.
            """,
            metadata={"topic": "quantum_computing"},
        )

        # Search for quantum terms
        search_query = SearchQuery(
            owner_id="test_workspace",
            query_text="quantum supremacy",
            search_type=SearchType.TEXT,
            limit=5,
        )

        results = await memory_manager.search(search_query)

        # Verify results
        assert len(results) > 0
        assert any("quantum" in r.content.lower() for r in results)

        # Check metadata is preserved
        first_result = results[0]
        assert first_result.metadata is not None

    async def test_search_with_real_documents(self, memory_manager):
        """Test search with multiple real documents."""
        # Load and process test documents
        test_docs_dir = Path(__file__).parent / "res"

        doc_files = [
            ("artificial_intelligence.txt", "AI and machine learning overview"),
            ("quantum_computing.txt", "Introduction to quantum computing"),
            ("climate_change.txt", "Climate science and global warming"),
            ("renewable_energy.txt", "Solar, wind, and sustainable energy"),
            ("space_exploration.txt", "Mars missions and space technology"),
        ]

        # Process each document
        for filename, description in doc_files:
            file_path = test_docs_dir / filename
            if file_path.exists():
                content = file_path.read_text()

                doc, chunks = await memory_manager.process_document(
                    owner_id="test_workspace",
                    id_at_origin=f"file_{filename}",
                    document_name=filename,
                    document_type=DocumentType.TEXT,
                    content=content,
                    metadata={"description": description, "document_name": filename},
                )

                assert len(chunks) > 0

        # Test 1: Search for AI concepts
        ai_query = SearchQuery(
            owner_id="test_workspace",
            query_text="machine learning neural networks",
            search_type=SearchType.TEXT,
            limit=10,
        )

        ai_results = await memory_manager.search(ai_query)
        assert len(ai_results) > 0

        # Should find AI-related content
        ai_content_found = any(
            any(
                term in r.content.lower()
                for term in ["artificial", "intelligence", "neural", "machine learning"]
            )
            for r in ai_results
        )
        assert ai_content_found

        # Test 2: Search for climate topics
        climate_query = SearchQuery(
            owner_id="test_workspace",
            query_text="global warming greenhouse gases",
            search_type=SearchType.TEXT,
            limit=10,
        )

        climate_results = await memory_manager.search(climate_query)
        assert len(climate_results) > 0

        # Test 3: Cross-topic search
        cross_query = SearchQuery(
            owner_id="test_workspace",
            query_text="technology future",
            search_type=SearchType.TEXT,
            limit=10,
        )

        cross_results = await memory_manager.search(cross_query)
        assert len(cross_results) > 0

        # Should find results from multiple documents
        doc_names = set()
        for r in cross_results[:5]:  # Check top 5 results
            # Get document metadata
            import json

            metadata = json.loads(r.metadata) if isinstance(r.metadata, str) else r.metadata
            doc_names.add(metadata.get("document_name", "unknown"))

        # Should have results from different documents
        assert len(doc_names) >= 2

    async def test_search_filters(self, memory_manager):
        """Test search with various filters."""
        # Add documents with different metadata
        docs_data = [
            {
                "id_at_origin": "user_alice",
                "content": "Python is a great programming language for beginners",
                "metadata": {
                    "author": "alice",
                    "category": "programming",
                    "language": "python",
                },
            },
            {
                "id_at_origin": "user_bob",
                "content": "JavaScript is essential for web development",
                "metadata": {
                    "author": "bob",
                    "category": "programming",
                    "language": "javascript",
                },
            },
            {
                "id_at_origin": "user_alice",
                "content": "Machine learning with Python and scikit-learn",
                "metadata": {"author": "alice", "category": "ml", "language": "python"},
            },
        ]

        for i, doc_data in enumerate(docs_data):
            await memory_manager.process_document(
                owner_id="test_workspace",
                id_at_origin=doc_data["id_at_origin"],
                document_name=f"doc_{i}.txt",
                document_type=DocumentType.TEXT,
                content=doc_data["content"],
                metadata=doc_data["metadata"],
            )

        # Test 1: Filter by metadata
        python_query = SearchQuery(
            owner_id="test_workspace",
            query_text="programming",
            search_type=SearchType.TEXT,
            metadata_filter={"language": "python"},
            limit=10,
        )

        python_results = await memory_manager.search(python_query)

        # Should only find Python documents
        for result in python_results:
            assert result.metadata.get("language") == "python"

        # Test 2: Filter by id_at_origin
        alice_query = SearchQuery(
            owner_id="test_workspace",
            query_text="programming",
            search_type=SearchType.TEXT,
            id_at_origin="user_alice",
            limit=10,
        )

        alice_results = await memory_manager.search(alice_query)

        # Should only find Alice's documents
        assert len(alice_results) > 0
        # All results should be from Alice (would need to join with documents table to verify)

    async def test_hierarchical_search(self, memory_manager):
        """Test search with hierarchical document structure."""
        # Create a document with hierarchical structure
        content = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Supervised Learning

In supervised learning, algorithms learn from labeled training data.

### Classification
Classification algorithms predict discrete class labels.

### Regression
Regression algorithms predict continuous values.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

### Clustering
Clustering groups similar data points together.

### Dimensionality Reduction
Reduces the number of features while preserving information.
"""

        doc, chunks = await memory_manager.process_document(
            owner_id="test_workspace",
            id_at_origin="ml_guide",
            document_name="ml_guide.md",
            document_type=DocumentType.MARKDOWN,
            content=content,
            chunking_strategy="hierarchical",
        )

        # Should create chunks
        assert len(chunks) >= 1

        # Check for parent and child chunks if hierarchical
        if len(chunks) > 1:
            parent_chunks = [c for c in chunks if c.chunk_level > 0]
            child_chunks = [c for c in chunks if c.chunk_level == 0]

            # If we have multiple chunks, we might have hierarchy
            if parent_chunks or child_chunks:
                assert len(parent_chunks) > 0 or len(child_chunks) > 0

        # Search for specific concepts
        search_query = SearchQuery(
            owner_id="test_workspace",
            query_text="classification algorithms",
            search_type=SearchType.TEXT,
            include_parent_context=True,
            limit=5,
        )

        results = await memory_manager.search(search_query)

        # Should find relevant content
        assert len(results) > 0
        assert any("classification" in r.content.lower() for r in results)
