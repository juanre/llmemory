import pytest

from llmemory import DocumentType


@pytest.mark.asyncio
async def test_parent_context_uses_hierarchy(memory_library):
    """Test parent context retrieval uses parent_chunk_id relationships."""

    memory = memory_library

    # Add hierarchical document with distinct parent and child content
    # Parent chunk will be large, child chunks will be smaller
    parent_content = "This is the parent section with important context. " * 50
    child1_content = "First child subsection with specific details. " * 30
    child2_content = "Second child subsection with more information. " * 30

    content = parent_content + "\n\n" + child1_content + "\n\n" + child2_content

    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test_hierarchical",
        document_name="hierarchical_test.txt",
        document_type=DocumentType.TEXT,
        content=content,
        chunking_strategy="hierarchical",
        generate_embeddings=False,  # Not testing embeddings
    )

    # Search for child chunk content using TEXT search
    # (vector search requires embeddings which we disabled)
    from uuid import UUID

    from llmemory.models import SearchType

    results = await memory.search(
        owner_id="test",
        query_text="specific details",
        search_type=SearchType.TEXT,
        include_parent_context=True,
        context_window=1,
        limit=10,  # Get multiple results to find a child chunk
    )

    assert len(results) > 0, "Should find at least one result"

    # Find a result that is a child chunk (has a parent_chunk_id)
    search_result = None
    expected_parent_id = None
    for result in results:
        # Check if this chunk has a parent
        check_query = """
            SELECT parent_chunk_id
            FROM {{tables.document_chunks}}
            WHERE chunk_id = $1
        """
        chunk_data = await memory._manager.db.db.fetch_one(check_query, UUID(str(result.chunk_id)))
        if chunk_data and chunk_data["parent_chunk_id"]:
            search_result = result
            expected_parent_id = chunk_data["parent_chunk_id"]
            break

    assert search_result is not None, "Should find at least one child chunk with a parent"

    # Should include parent context
    assert search_result.parent_chunks is not None, "Parent chunks should be included"
    assert len(search_result.parent_chunks) > 0, "Should have at least one parent chunk"

    # Verify we got the hierarchical parent (via parent_chunk_id), not just adjacent
    # The parent should contain the parent section text
    parent = search_result.parent_chunks[0]

    # Check that parent has parent_chunk_id relationship (chunk_level should be higher)
    # In hierarchical chunking, parent chunks have lower chunk_level numbers
    assert parent.chunk_level is not None, "Parent should have chunk_level"

    # Parent should contain parent content
    assert (
        "parent section with important context" in parent.content.lower()
    ), f"Parent should contain hierarchical parent content, got: {parent.content[:200]}"

    # Most importantly: verify the returned chunk's parent_chunk_id matches our parent
    # This proves we're using hierarchical relationships, not just adjacency
    # asyncpg returns UUID objects, no need to convert
    if isinstance(expected_parent_id, str):
        expected_parent_id = UUID(expected_parent_id)
    assert (
        parent.chunk_id == expected_parent_id
    ), f"Parent chunk should be the one referenced by parent_chunk_id ({expected_parent_id}), not just adjacent ({parent.chunk_id})"
