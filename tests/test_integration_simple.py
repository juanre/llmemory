"""Simple integration test to verify the setup works."""

import pytest
from llmemory import DocumentType
from llmemory.models import SearchType


@pytest.mark.asyncio
async def test_simple_integration(memory_library):
    """Test basic integration with database."""
    # Use the fixture which handles database setup
    memory = memory_library

    # Process a simple document
    doc = await memory.add_document(
        owner_id="test_workspace",
        id_at_origin="test_user",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="This is a test document about artificial intelligence and machine learning.",
    )

    assert doc is not None

    # Try a simple search
    results = await memory.search(
        owner_id="test_workspace",
        query_text="artificial intelligence",
        search_type=SearchType.TEXT,
        limit=5,
    )

    assert isinstance(results, list)


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    # This test is now async
    print("Run with pytest: pytest tests/test_integration_simple.py")
