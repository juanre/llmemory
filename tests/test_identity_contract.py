"""Tests for the archive-protocol identity contract.

Validates that llmemory correctly implements idempotent upsert behavior
using (owner_id, id_at_origin) as the natural key per the SOT:
- docs-sot/source-of-truth/llmemory-search.md
- docs-sot/source-of-truth/archive-protocol.md
"""

from datetime import datetime

import pytest

from llmemory.models import DocumentType


@pytest.mark.asyncio
class TestIdentityContract:
    """Test archive-protocol identity contract: (owner_id, id_at_origin)."""

    async def test_add_document_creates_new(self, memory_manager):
        """Test that add_document creates a new document."""
        doc = await memory_manager.add_document(
            owner_id="jro",  # Archive-protocol entity
            id_at_origin="doc-2024-001",  # Archive-protocol document_id
            document_name="invoice.pdf",
            document_type=DocumentType.PDF,
        )

        assert doc.owner_id == "jro"
        assert doc.id_at_origin == "doc-2024-001"
        assert doc.document_name == "invoice.pdf"
        assert doc.created_at is not None
        assert doc.updated_at is not None

    async def test_add_document_upsert_preserves_document_id(self, memory_manager):
        """Test that re-adding same (owner_id, id_at_origin) preserves document_id."""
        # First add
        doc1 = await memory_manager.add_document(
            owner_id="jro",
            id_at_origin="doc-2024-002",
            document_name="contract_v1.pdf",
            document_type=DocumentType.PDF,
        )
        original_document_id = doc1.document_id

        # Re-add with same identity but different name
        doc2 = await memory_manager.add_document(
            owner_id="jro",
            id_at_origin="doc-2024-002",
            document_name="contract_v2.pdf",  # Updated name
            document_type=DocumentType.PDF,
        )

        # Document ID should be preserved
        assert doc2.document_id == original_document_id
        # Name should be updated
        assert doc2.document_name == "contract_v2.pdf"

    async def test_add_document_upsert_updates_metadata(self, memory_manager):
        """Test that upsert updates metadata while preserving document_id."""
        # First add with initial metadata
        doc1 = await memory_manager.add_document(
            owner_id="tsm",
            id_at_origin="doc-2024-003",
            document_name="report.pdf",
            document_type=DocumentType.PDF,
            metadata={"version": 1, "status": "draft"},
        )

        # Re-add with updated metadata
        doc2 = await memory_manager.add_document(
            owner_id="tsm",
            id_at_origin="doc-2024-003",
            document_name="report.pdf",
            document_type=DocumentType.PDF,
            metadata={"version": 2, "status": "final"},
        )

        # Document ID preserved
        assert doc2.document_id == doc1.document_id
        # Metadata updated
        assert doc2.metadata["version"] == 2
        assert doc2.metadata["status"] == "final"

    async def test_different_id_at_origin_creates_different_documents(self, memory_manager):
        """Test that different id_at_origin creates separate documents."""
        doc1 = await memory_manager.add_document(
            owner_id="jro",
            id_at_origin="doc-2024-004",
            document_name="invoice_jan.pdf",
            document_type=DocumentType.PDF,
        )

        doc2 = await memory_manager.add_document(
            owner_id="jro",
            id_at_origin="doc-2024-005",  # Different id_at_origin
            document_name="invoice_feb.pdf",
            document_type=DocumentType.PDF,
        )

        # Should be different documents
        assert doc1.document_id != doc2.document_id

    async def test_different_owner_id_creates_different_documents(self, memory_manager):
        """Test that different owner_id creates separate documents."""
        doc1 = await memory_manager.add_document(
            owner_id="jro",
            id_at_origin="shared-doc-001",
            document_name="shared.pdf",
            document_type=DocumentType.PDF,
        )

        doc2 = await memory_manager.add_document(
            owner_id="tsm",  # Different owner
            id_at_origin="shared-doc-001",  # Same id_at_origin
            document_name="shared.pdf",
            document_type=DocumentType.PDF,
        )

        # Should be different documents (different owner)
        assert doc1.document_id != doc2.document_id


@pytest.mark.asyncio
class TestProcessDocumentIdempotent:
    """Test idempotent re-indexing via process_document."""

    async def test_process_document_creates_chunks(self, memory_manager):
        """Test that process_document creates document and chunks."""
        doc, chunks = await memory_manager.process_document(
            owner_id="gsk",
            id_at_origin="doc-2024-010",
            document_name="notes.md",
            document_type=DocumentType.MARKDOWN,
            content="# Notes\n\nSome notes here.",
        )

        assert doc.owner_id == "gsk"
        assert doc.id_at_origin == "doc-2024-010"
        assert len(chunks) > 0

    async def test_reindex_preserves_document_id_replaces_chunks(self, memory_manager):
        """Test that re-indexing preserves document_id but replaces chunks."""
        # First indexing
        doc1, chunks1 = await memory_manager.process_document(
            owner_id="jro",
            id_at_origin="doc-2024-011",
            document_name="guide.md",
            document_type=DocumentType.MARKDOWN,
            content="# Guide v1\n\nOriginal content.",
        )
        original_document_id = doc1.document_id
        original_chunk_ids = [c.chunk_id for c in chunks1]

        # Re-index with updated content
        doc2, chunks2 = await memory_manager.process_document(
            owner_id="jro",
            id_at_origin="doc-2024-011",  # Same identity
            document_name="guide.md",
            document_type=DocumentType.MARKDOWN,
            content="# Guide v2\n\nUpdated content with more details.\n\n## New Section\n\nNew stuff.",
        )

        # Document ID preserved
        assert doc2.document_id == original_document_id

        # Chunks should be new (different from original)
        new_chunk_ids = [c.chunk_id for c in chunks2]
        assert set(new_chunk_ids).isdisjoint(set(original_chunk_ids))

    async def test_reindex_no_duplicates(self, memory_manager):
        """Test that re-indexing multiple times doesn't create duplicates."""
        identity = ("jro", "doc-2024-012")

        # Index the same document 3 times with different content
        for version in range(1, 4):
            doc, chunks = await memory_manager.process_document(
                owner_id=identity[0],
                id_at_origin=identity[1],
                document_name=f"report_v{version}.md",
                document_type=DocumentType.MARKDOWN,
                content=f"# Report Version {version}\n\nContent for version {version}.",
            )

        # Query to check there's only one document with this identity
        query = """
        SELECT COUNT(*) as count FROM {{tables.documents}}
        WHERE owner_id = $1 AND id_at_origin = $2
        """
        result = await memory_manager.db.db_manager.fetch_one(query, identity[0], identity[1])

        assert result["count"] == 1, "Should have exactly one document, not duplicates"

    async def test_reindex_updates_updated_at(self, memory_manager):
        """Test that re-indexing updates the updated_at timestamp."""
        # First indexing
        doc1, _ = await memory_manager.process_document(
            owner_id="tsm",
            id_at_origin="doc-2024-013",
            document_name="memo.txt",
            document_type=DocumentType.TEXT,
            content="Initial memo content.",
        )

        # Small delay to ensure different timestamp
        import asyncio
        await asyncio.sleep(0.1)

        # Re-index
        doc2, _ = await memory_manager.process_document(
            owner_id="tsm",
            id_at_origin="doc-2024-013",
            document_name="memo.txt",
            document_type=DocumentType.TEXT,
            content="Updated memo content.",
        )

        # created_at should be same, updated_at should be different
        assert doc2.created_at == doc1.created_at
        assert doc2.updated_at > doc1.updated_at


@pytest.mark.asyncio
class TestDeleteDocumentChunks:
    """Test chunk deletion for re-indexing."""

    async def test_delete_document_chunks(self, memory_manager):
        """Test that delete_document_chunks removes all chunks."""
        # Create document with chunks
        doc, chunks = await memory_manager.process_document(
            owner_id="jro",
            id_at_origin="doc-2024-020",
            document_name="test.md",
            document_type=DocumentType.MARKDOWN,
            content="# Test\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2.",
        )

        assert len(chunks) > 0

        # Count actual chunks in database using direct SQL
        initial_count = await memory_manager.db.count_chunks(str(doc.document_id))
        assert initial_count > 0

        # Delete chunks
        deleted_count = await memory_manager.delete_document_chunks(doc.document_id)

        # Should have deleted all chunks in database
        assert deleted_count == initial_count

        # Verify chunks are gone
        remaining_count = await memory_manager.db.count_chunks(str(doc.document_id))
        assert remaining_count == 0

    async def test_delete_document_chunks_preserves_document(self, memory_manager):
        """Test that delete_document_chunks preserves the document record."""
        # Create document with chunks
        doc, chunks = await memory_manager.process_document(
            owner_id="tsm",
            id_at_origin="doc-2024-021",
            document_name="preserve.md",
            document_type=DocumentType.MARKDOWN,
            content="# Preserve\n\nThis document should remain.",
        )

        assert len(chunks) > 0

        # Delete chunks
        await memory_manager.delete_document_chunks(doc.document_id)

        # Document should still exist
        exists = await memory_manager.db.document_exists(str(doc.document_id))
        assert exists, "Document should still exist after chunk deletion"

        # But chunks should be gone
        chunk_count = await memory_manager.db.count_chunks(str(doc.document_id))
        assert chunk_count == 0
