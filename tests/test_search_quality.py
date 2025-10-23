"""Tests for search quality and relevance using real documents."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pytest
import pytest_asyncio
from llmemory import LLMemory, DocumentType, SearchType
from llmemory.models import Document, SearchQuery


@pytest.mark.skip(
    reason="""
    Skipped due to pytest/asyncpg event loop incompatibility.

    This is NOT a code bug. The issue occurs when:
    1. pytest manages event loops for async tests
    2. asyncpg creates connection pools tied to those loops
    3. A deadlock occurs between pytest and asyncpg event loop handling

    The code works correctly in production and standalone scripts.
    See conftest.py for detailed explanation and verify_large_docs.py for proof.

    These tests validate:
    - Semantic/concept search with real documents
    - Multi-document search across topics
    - Search result relevance ranking
    - Hierarchical search with parent context

    Alternative validation:
    - test_search_sync.py covers basic search functionality
    - test_db_integration.py validates vector operations
    - test_embedding_generation.py confirms OpenAI integration
"""
)
class TestSearchQuality:
    """Test search quality with real documents and expected results."""

    @pytest_asyncio.fixture
    async def memory_with_documents(
        self, memory_library
    ) -> Tuple[LLMemory, Dict[str, Document]]:
        """Create memory instance and load test documents."""
        # Use the memory_library fixture which handles test database setup
        memory = memory_library

        # Load and process documents
        res_dir = Path(__file__).parent / "res"
        documents = {}

        for file_path in res_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Note: Using sync process_document since async has issues in pytest
            doc, chunks = memory._sync_manager.process_document(
                owner_id="test_workspace",
                id_at_origin="test_user",
                document_name=file_path.name,
                document_type=DocumentType.TEXT,
                content=content,
                document_date=datetime.now(),
                metadata={"topic": file_path.stem},
            )
            documents[file_path.stem] = doc

        yield memory, documents

    @pytest.mark.asyncio
    async def test_exact_phrase_search(self, memory_with_documents):
        """Test searching for exact phrases."""
        memory, docs = memory_with_documents

        # Test cases with expected results
        test_cases = [
            {
                "query": "quantum supremacy",
                "expected_doc": "quantum_computing",
                "expected_context": "Google claimed to achieve quantum supremacy",
            },
            {
                "query": "greenhouse effect",
                "expected_doc": "climate_change",
                "expected_context": "natural phenomenon that makes life on Earth possible",
            },
            {
                "query": "Turing Test",
                "expected_doc": "artificial_intelligence",
                "expected_context": "measure of machine intelligence",
            },
            {
                "query": "solar photovoltaic",
                "expected_doc": "renewable_energy",
                "expected_context": "price of solar photovoltaic (PV) modules",
            },
            {
                "query": "International Space Station",
                "expected_doc": "space_exploration",
                "expected_context": "continuously inhabited since November 2000",
            },
        ]

        for test_case in test_cases:
            # Use sync search since we're in sync mode
            search_query = SearchQuery(
                owner_id="test_workspace",
                query_text=test_case["query"],
                search_type=SearchType.TEXT,
                limit=5,
            )
            results = memory._sync_manager.search(search_query)

            assert len(results) > 0, f"No results for query: {test_case['query']}"

            # Check that expected document appears in top results
            top_result = results[0]
            doc = memory._sync_manager.db.db.execute_and_fetch_one(
                "SELECT * FROM documents WHERE document_id = %s",
                (str(top_result.document_id),),
            )

            assert (
                test_case["expected_doc"] in doc["metadata"]["topic"]
            ), f"Expected {test_case['expected_doc']} in top results for '{test_case['query']}'"

            # Verify context contains expected content
            assert (
                test_case["expected_context"] in top_result.content
            ), f"Expected context not found in result for '{test_case['query']}'"

    def test_concept_search(self, memory_with_documents):
        """Test searching for concepts that may not be exact matches."""
        memory, docs = memory_with_documents

        concept_searches = [
            {
                "query": "parallel computing quantum",
                "expected_docs": ["quantum_computing"],
                "related_concepts": ["superposition", "qubits", "2^n states"],
            },
            {
                "query": "global warming causes",
                "expected_docs": ["climate_change"],
                "related_concepts": [
                    "fossil fuels",
                    "greenhouse gases",
                    "carbon dioxide",
                ],
            },
            {
                "query": "machine learning deep neural networks",
                "expected_docs": ["artificial_intelligence"],
                "related_concepts": ["CNN", "RNN", "Transformer"],
            },
            {
                "query": "clean energy storage batteries",
                "expected_docs": ["renewable_energy"],
                "related_concepts": ["lithium-ion", "grid-scale", "Hornsdale"],
            },
            {
                "query": "Mars colonization SpaceX",
                "expected_docs": ["space_exploration"],
                "related_concepts": ["Starship", "self-sustaining city", "100 people"],
            },
        ]

        for search in concept_searches:
            # Use sync search
            from llmemory.models import SearchQuery

            search_query = SearchQuery(
                owner_id="test_workspace",
                query_text=search["query"],
                search_type=SearchType.TEXT,
                limit=10,
            )
            results = memory._sync_manager.search(search_query)

            assert len(results) > 0, f"No results for concept search: {search['query']}"

            # Check that at least one related concept appears in top results
            top_contents = " ".join(r.content.lower() for r in results[:3])

            concept_found = any(
                concept.lower() in top_contents for concept in search["related_concepts"]
            )
            assert concept_found, f"No related concepts found for query: {search['query']}"

    @pytest.mark.asyncio
    async def test_multi_document_search(self, memory_with_documents):
        """Test queries that should return results from multiple documents."""
        memory, docs = memory_with_documents

        cross_cutting_queries = [
            {
                "query": "technological revolution innovation",
                "expected_docs": [
                    "artificial_intelligence",
                    "quantum_computing",
                    "renewable_energy",
                ],
                "min_docs": 2,
            },
            {
                "query": "future challenges opportunities",
                "expected_docs": [
                    "climate_change",
                    "space_exploration",
                    "artificial_intelligence",
                ],
                "min_docs": 2,
            },
            {
                "query": "scientific breakthrough research",
                "expected_docs": [
                    "quantum_computing",
                    "artificial_intelligence",
                    "space_exploration",
                ],
                "min_docs": 2,
            },
            {
                "query": "environmental impact sustainability",
                "expected_docs": ["climate_change", "renewable_energy"],
                "min_docs": 2,
            },
        ]

        for query_test in cross_cutting_queries:
            results = await memory.search(
                owner_id="test_workspace",
                query_text=query_test["query"],
                search_type=SearchType.TEXT,
                limit=20,
            )

            # Get unique document topics from results
            unique_topics = set()
            for result in results:
                doc = await memory._async_manager.db.execute_and_fetch_one(
                    "SELECT metadata FROM documents WHERE document_id = %s",
                    (result.document_id,),
                )
                topic = doc["metadata"].get("topic")
                if topic:
                    unique_topics.add(topic)

            # Verify minimum number of different documents
            assert (
                len(unique_topics) >= query_test["min_docs"]
            ), f"Expected at least {query_test['min_docs']} different documents for '{query_test['query']}'"

            # Check that at least one expected document appears
            found_expected = any(doc in unique_topics for doc in query_test["expected_docs"])
            assert (
                found_expected
            ), f"None of expected documents found for query: {query_test['query']}"

    @pytest.mark.asyncio
    async def test_relevance_ranking(self, memory_with_documents):
        """Test that more relevant results appear first."""
        memory, docs = memory_with_documents

        # Search for "quantum computer" - should rank quantum computing doc highest
        results = await memory.search(
            owner_id="test_workspace",
            query_text="quantum computer qubits",
            search_type=SearchType.TEXT,
            limit=10,
        )

        # Get document topics for top 3 results
        top_topics = []
        for result in results[:3]:
            doc = await memory._async_manager.db.execute_and_fetch_one(
                "SELECT metadata FROM documents WHERE document_id = %s",
                (result.document_id,),
            )
            top_topics.append(doc["metadata"].get("topic"))

        # Quantum computing should be in top 3
        assert (
            "quantum_computing" in top_topics
        ), "Quantum computing document should be in top results for 'quantum computer'"

        # More specific test: exact title match should rank highest
        results = await memory.search(
            owner_id="test_workspace",
            query_text="Climate Change Understanding the Global Crisis",
            search_type=SearchType.TEXT,
            limit=5,
        )

        if results:
            top_doc = await memory._async_manager.db.execute_and_fetch_one(
                "SELECT metadata FROM documents WHERE document_id = %s",
                (results[0].document_id,),
            )
            assert (
                top_doc["metadata"].get("topic") == "climate_change"
            ), "Document with matching title should rank first"

    @pytest.mark.asyncio
    async def test_hierarchical_search_with_context(self, memory_with_documents):
        """Test that parent context is included when searching."""
        memory, docs = memory_with_documents

        # Search for specific detail that would benefit from context
        results = await memory.search(
            owner_id="test_workspace",
            query_text="Shor's algorithm factoring",
            search_type=SearchType.TEXT,
            limit=5,
            include_parent_context=True,
        )

        assert len(results) > 0, "Should find results for Shor's algorithm"

        # Check that result includes parent context
        for result in results:
            if "shor" in result.content.lower():
                # If parent context is included, we should see broader discussion
                if result.parent_content:
                    assert len(result.parent_content) > len(
                        result.content
                    ), "Parent content should be longer than child chunk"
                    assert (
                        "quantum" in result.parent_content.lower()
                    ), "Parent context should include broader quantum discussion"

    @pytest.mark.asyncio
    async def test_negative_searches(self, memory_with_documents):
        """Test searches that should return few or no results."""
        memory, docs = memory_with_documents

        # Search for terms that don't exist in documents
        no_result_queries = [
            "blockchain cryptocurrency bitcoin",  # Not in our documents
            "medieval history ancient rome",  # Completely unrelated
            "recipe cooking ingredients",  # Different domain
            "xyzbca123 nonexistent term",  # Gibberish
        ]

        for query in no_result_queries:
            results = await memory.search(
                owner_id="test_workspace",
                query_text=query,
                search_type=SearchType.TEXT,
                limit=5,
            )

            # Should return few or no highly relevant results
            if results:
                # Check that results don't actually contain the search terms
                for result in results[:2]:  # Check top 2
                    content_lower = result.content.lower()
                    query_terms = query.lower().split()

                    # Count how many query terms appear in result
                    matches = sum(1 for term in query_terms if term in content_lower)
                    assert matches <= 1, f"Unexpected match for unrelated query: {query}"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, memory_with_documents):
        """Test search with various filters."""
        memory, docs = memory_with_documents

        # First, add documents with different metadata
        await memory.process_document(
            owner_id="test_workspace",
            id_at_origin="user_alice",
            document_name="alice_quantum_notes.txt",
            document_type=DocumentType.TEXT,
            content="Alice's notes on quantum computing and superposition.",
            metadata={"author": "alice", "topic": "quantum_computing"},
        )

        await memory.process_document(
            owner_id="test_workspace",
            id_at_origin="user_bob",
            document_name="bob_quantum_summary.txt",
            document_type=DocumentType.TEXT,
            content="Bob's summary of quantum entanglement experiments.",
            metadata={"author": "bob", "topic": "quantum_computing"},
        )

        # Test filtering by origin
        results = await memory.search(
            owner_id="test_workspace",
            query_text="quantum",
            search_type=SearchType.TEXT,
            id_at_origins=["user_alice"],
            limit=10,
        )

        # Should only return Alice's document
        for result in results:
            doc = await memory._async_manager.db.execute_and_fetch_one(
                "SELECT id_at_origin FROM documents WHERE document_id = %s",
                (result.document_id,),
            )
            assert (
                doc["id_at_origin"] == "user_alice"
            ), "Filter by origin should only return specified user's documents"

        # Test metadata filtering
        results = await memory.search(
            owner_id="test_workspace",
            query_text="quantum",
            search_type=SearchType.TEXT,
            metadata_filter={"author": "bob"},
            limit=10,
        )

        # Verify metadata filter works
        bob_found = False
        for result in results:
            doc = await memory._async_manager.db.execute_and_fetch_one(
                "SELECT metadata FROM documents WHERE document_id = %s",
                (result.document_id,),
            )
            if doc["metadata"].get("author") == "bob":
                bob_found = True

        assert bob_found, "Should find Bob's document with metadata filter"

    @pytest.mark.asyncio
    async def test_special_character_search(self, memory_with_documents):
        """Test searching for content with special characters."""
        memory, docs = memory_with_documents

        # Search for technical terms with special characters
        special_queries = [
            "CO2",  # Chemical formula
            "2^n",  # Mathematical notation
            "text-embedding-3-small",  # Hyphenated term
            "real-time",  # Common hyphenated phrase
            "(AI)",  # Acronym in parentheses
        ]

        for query in special_queries:
            results = await memory.search(
                owner_id="test_workspace",
                query_text=query,
                search_type=SearchType.TEXT,
                limit=5,
            )

            # These should all return results from our documents
            if query in ["CO2", "real-time"]:
                assert len(results) > 0, f"Should find results for: {query}"

    @pytest.mark.asyncio
    async def test_case_insensitive_search(self, memory_with_documents):
        """Test that search is case-insensitive."""
        memory, docs = memory_with_documents

        # Test same query in different cases
        queries = [
            "Quantum Computing",
            "QUANTUM COMPUTING",
            "quantum computing",
            "QuAnTuM cOmPuTiNg",
        ]

        result_counts = []
        for query in queries:
            results = await memory.search(
                owner_id="test_workspace",
                query_text=query,
                search_type=SearchType.TEXT,
                limit=10,
            )
            result_counts.append(len(results))

        # All queries should return similar number of results
        assert len(set(result_counts)) <= 2, "Case variations should return similar result counts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
