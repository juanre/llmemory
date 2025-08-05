"""Simple test to verify chunking works."""

from llmemory.chunking import HierarchicalChunker
from llmemory.models import DocumentType


def test_basic_chunking():
    """Test basic chunking functionality."""
    # Create chunker with default config
    chunker = HierarchicalChunker()

    # Simple test text
    text = """
    Artificial Intelligence is transforming the world. Machine learning algorithms
    are becoming more sophisticated every day. Deep learning models can now
    understand and generate human-like text with remarkable accuracy.

    The applications of AI are vast and growing. From healthcare to finance,
    from transportation to entertainment, AI is making its mark everywhere.
    Companies are investing billions in AI research and development.

    However, with great power comes great responsibility. We must ensure that
    AI systems are developed ethically and with proper safeguards. The future
    of AI depends on how we shape it today.
    """

    # Chunk the document
    chunks = chunker.chunk_document(
        text=text,
        document_id="test_doc_1",
        document_type=DocumentType.TEXT,
        base_metadata={"source": "test"},
    )

    # Basic assertions
    assert len(chunks) > 0, "Should create at least one chunk"

    # Check chunk properties
    for chunk in chunks:
        assert chunk.content, "Chunk should have content"
        assert chunk.document_id == "test_doc_1"
        assert chunk.metadata.get("source") == "test"

        # Check chunk type
        chunk_type = chunk.metadata.get("chunk_type")
        assert chunk_type in ["parent", "child"], f"Invalid chunk type: {chunk_type}"

        if chunk_type == "child":
            # Child chunks should have parent context
            assert "parent_context" in chunk.metadata

    print(f"Created {len(chunks)} chunks")
    parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    print(f"Parents: {len(parent_chunks)}, Children: {len(child_chunks)}")


if __name__ == "__main__":
    test_basic_chunking()
