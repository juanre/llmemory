"""Tests for new aword-memory API methods."""

import asyncio
from datetime import datetime, timedelta

import pytest
from llmemory import (
    LLMemory,
    DeleteResult,
    DocumentListResult,
    DocumentNotFoundError,
    DocumentType,
    DocumentWithChunks,
    OwnerStatistics,
    SearchResultWithDocuments,
)
from llmemory.models import Document, DocumentChunk


class TestDocumentListingAPI:
    """Test the list_documents API method."""

    @pytest.mark.asyncio
    async def test_list_documents_basic(self, memory_library: LLMemory):
        """Test basic document listing."""
        # Add test documents
        owner_id = "test_owner"
        docs = []

        for i in range(5):
            result = await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=DocumentType.TEXT,
                content=f"Content for document {i}",
            )
            docs.append(result.document)

        # List documents
        list_result = await memory_library.list_documents(owner_id=owner_id, limit=10)

        assert isinstance(list_result, DocumentListResult)
        assert list_result.total == 5
        assert len(list_result.documents) == 5
        assert list_result.limit == 10
        assert list_result.offset == 0

        # Verify document properties
        for doc in list_result.documents:
            assert isinstance(doc, Document)
            assert doc.owner_id == owner_id
            assert doc.document_name.startswith("Document")

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, memory_library: LLMemory):
        """Test document listing with pagination."""
        owner_id = "test_owner_paginated"

        # Add 10 documents
        for i in range(10):
            await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i:02d}",
                document_type=DocumentType.TEXT,
                content=f"Content {i}",
            )

        # Test pagination - page 1
        page1 = await memory_library.list_documents(owner_id=owner_id, limit=3, offset=0)

        assert page1.total == 10
        assert len(page1.documents) == 3
        assert page1.limit == 3
        assert page1.offset == 0

        # Test pagination - page 2
        page2 = await memory_library.list_documents(owner_id=owner_id, limit=3, offset=3)

        assert page2.total == 10
        assert len(page2.documents) == 3
        assert page2.offset == 3

        # Ensure different documents
        page1_ids = {str(doc.document_id) for doc in page1.documents}
        page2_ids = {str(doc.document_id) for doc in page2.documents}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_documents_by_type(self, memory_library: LLMemory):
        """Test filtering documents by type."""
        owner_id = "test_owner_types"

        # Add documents of different types
        types = [
            DocumentType.TEXT,
            DocumentType.MARKDOWN,
            DocumentType.PDF,
            DocumentType.TEXT,
        ]
        for i, doc_type in enumerate(types):
            await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=doc_type,
                content=f"Content {i}",
            )

        # Filter by TEXT type
        text_docs = await memory_library.list_documents(
            owner_id=owner_id, document_type=DocumentType.TEXT
        )

        assert text_docs.total == 2
        assert all(doc.document_type == DocumentType.TEXT for doc in text_docs.documents)

        # Filter by MARKDOWN type
        md_docs = await memory_library.list_documents(
            owner_id=owner_id, document_type=DocumentType.MARKDOWN
        )

        assert md_docs.total == 1
        assert md_docs.documents[0].document_type == DocumentType.MARKDOWN

    @pytest.mark.asyncio
    async def test_list_documents_with_metadata_filter(self, memory_library: LLMemory):
        """Test filtering documents by metadata."""
        owner_id = "test_owner_metadata"

        # Add documents with metadata
        metadata_values = [
            {"category": "finance", "priority": "high"},
            {"category": "finance", "priority": "low"},
            {"category": "tech", "priority": "high"},
            {"category": "tech", "priority": "medium"},
        ]

        for i, metadata in enumerate(metadata_values):
            await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=DocumentType.TEXT,
                content=f"Content {i}",
                metadata=metadata,
            )

        # Filter by category=finance
        finance_docs = await memory_library.list_documents(
            owner_id=owner_id, metadata_filter={"category": "finance"}
        )

        assert finance_docs.total == 2
        assert all(doc.metadata["category"] == "finance" for doc in finance_docs.documents)

        # Filter by multiple metadata fields
        high_priority_finance = await memory_library.list_documents(
            owner_id=owner_id,
            metadata_filter={"category": "finance", "priority": "high"},
        )

        assert high_priority_finance.total == 1
        assert high_priority_finance.documents[0].metadata["priority"] == "high"

    @pytest.mark.asyncio
    async def test_list_documents_ordering(self, memory_library: LLMemory):
        """Test document ordering options."""
        owner_id = "test_owner_ordering"

        # Add documents with different dates
        base_date = datetime.now()
        for i in range(3):
            await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {chr(65+i)}",  # A, B, C
                document_type=DocumentType.TEXT,
                content=f"Content {i}",
                document_date=base_date - timedelta(days=i),
            )
            await asyncio.sleep(0.1)  # Ensure different created_at times

        # Order by created_at DESC (default)
        by_created_desc = await memory_library.list_documents(
            owner_id=owner_id, order_by="created_at", order_desc=True
        )

        # Last created should be first
        assert by_created_desc.documents[0].document_name == "Document C"

        # Order by document_name ASC
        by_name_asc = await memory_library.list_documents(
            owner_id=owner_id, order_by="document_name", order_desc=False
        )

        assert by_name_asc.documents[0].document_name == "Document A"
        assert by_name_asc.documents[1].document_name == "Document B"
        assert by_name_asc.documents[2].document_name == "Document C"


class TestDocumentRetrievalAPI:
    """Test the get_document API method."""

    @pytest.mark.asyncio
    async def test_get_document_basic(self, memory_library: LLMemory):
        """Test basic document retrieval."""
        # Add a document
        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc",
            document_name="Test Document",
            document_type=DocumentType.TEXT,
            content="This is a test document with some content.",
            metadata={"key": "value"},
        )

        doc_id = result.document.document_id

        # Retrieve the document
        doc_with_chunks = await memory_library.get_document(doc_id)

        assert isinstance(doc_with_chunks, DocumentWithChunks)
        assert doc_with_chunks.document.document_id == doc_id
        assert doc_with_chunks.document.document_name == "Test Document"
        # Check that our metadata is preserved (language detection adds extra fields)
        assert doc_with_chunks.document.metadata["key"] == "value"
        assert "language" in doc_with_chunks.document.metadata  # Language detection was performed
        assert doc_with_chunks.chunk_count > 0
        assert doc_with_chunks.chunks is None  # Not requested

    @pytest.mark.asyncio
    async def test_get_document_with_chunks(self, memory_library: LLMemory):
        """Test document retrieval with chunks."""
        # Add a document with longer content for multiple chunks
        content = """
        # Chapter 1: Introduction
        This is the introduction to our document. It contains important information.

        # Chapter 2: Main Content
        The main content goes here. It's much longer and more detailed.
        We need to ensure it creates multiple chunks.

        # Chapter 3: Conclusion
        This is the conclusion of our document.
        """

        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc_chunks",
            document_name="Multi-chunk Document",
            document_type=DocumentType.MARKDOWN,
            content=content,
            chunking_strategy="hierarchical",
        )

        # Retrieve with chunks
        doc_with_chunks = await memory_library.get_document(
            result.document.document_id, include_chunks=True
        )

        assert doc_with_chunks.chunks is not None
        assert len(doc_with_chunks.chunks) > 0
        assert doc_with_chunks.chunk_count == len(doc_with_chunks.chunks)

        # Verify chunk properties
        for chunk in doc_with_chunks.chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == result.document.document_id
            assert chunk.content != ""
            assert chunk.chunk_index >= 0

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, memory_library: LLMemory):
        """Test retrieving non-existent document."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(DocumentNotFoundError):
            await memory_library.get_document(fake_id)

    @pytest.mark.asyncio
    async def test_get_document_string_uuid(self, memory_library: LLMemory):
        """Test retrieving document with string UUID."""
        # Add a document
        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc",
            document_name="Test Document",
            document_type=DocumentType.TEXT,
            content="Content",
        )

        # Retrieve using string UUID
        doc_with_chunks = await memory_library.get_document(str(result.document.document_id))

        assert doc_with_chunks.document.document_id == result.document.document_id


class TestEnhancedSearchAPI:
    """Test the search_with_documents API method."""

    @pytest.mark.asyncio
    async def test_search_with_documents_basic(self, memory_library_with_embeddings: LLMemory):
        """Test enhanced search with document metadata."""
        memory = memory_library_with_embeddings

        # Search for Python content
        results = await memory.search_with_documents(
            owner_id="test_workspace", query_text="Python programming", limit=5
        )

        assert isinstance(results, SearchResultWithDocuments)
        assert len(results.results) > 0
        assert results.total > 0

        # Check enriched results
        for result in results.results:
            assert result.document_name != ""
            assert result.document_type != ""
            assert isinstance(result.document_metadata, dict)

            # Verify it found Python-related content
            if "python" in result.content.lower():
                assert result.score > 0

    @pytest.mark.asyncio
    async def test_search_without_document_metadata(
        self, memory_library_with_embeddings: LLMemory
    ):
        """Test search without document metadata enrichment."""
        memory = memory_library_with_embeddings

        results = await memory.search_with_documents(
            owner_id="test_workspace",
            query_text="machine learning",
            include_document_metadata=False,
        )

        # Should still get results but without document metadata
        assert len(results.results) > 0
        for result in results.results:
            assert result.document_name == ""
            assert result.document_type == ""
            assert result.document_metadata == {}

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, memory_library: LLMemory):
        """Test search with chunk metadata filtering."""
        owner_id = "test_search_metadata"

        # Add documents with specific metadata
        await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="doc1",
            document_name="Technical Doc",
            document_type=DocumentType.TEXT,
            content="Python is great for data science",
            metadata={"category": "technical"},
        )

        await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="doc2",
            document_name="Business Doc",
            document_type=DocumentType.TEXT,
            content="Python helps business analytics",
            metadata={"category": "business"},
        )

        # Search with metadata filter
        results = await memory_library.search_with_documents(
            owner_id=owner_id,
            query_text="Python",
            metadata_filter={"category": "technical"},
        )

        # Should only find technical documents
        assert all(
            result.document_metadata.get("category") == "technical" for result in results.results
        )


class TestStatisticsAPI:
    """Test the get_statistics API method."""

    @pytest.mark.asyncio
    async def test_get_statistics_basic(self, memory_library: LLMemory):
        """Test basic statistics retrieval."""
        owner_id = "test_stats_owner"

        # Add various documents
        doc_types = [
            DocumentType.TEXT,
            DocumentType.MARKDOWN,
            DocumentType.PDF,
            DocumentType.TEXT,
        ]
        for i, doc_type in enumerate(doc_types):
            await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=doc_type,
                content=f"Content for document {i} " * 20,  # Make it longer
            )

        # Get statistics
        stats = await memory_library.get_statistics(owner_id)

        assert isinstance(stats, OwnerStatistics)
        assert stats.document_count == 4
        assert stats.chunk_count > 0
        assert stats.total_size_bytes > 0
        assert stats.document_type_breakdown is None  # Not requested
        assert stats.created_date_range is not None

    @pytest.mark.asyncio
    async def test_get_statistics_with_breakdown(self, memory_library: LLMemory):
        """Test statistics with document type breakdown."""
        owner_id = "test_stats_breakdown"

        # Add documents of different types
        await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="doc1",
            document_name="Text 1",
            document_type=DocumentType.TEXT,
            content="Content",
        )

        await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="doc2",
            document_name="Text 2",
            document_type=DocumentType.TEXT,
            content="Content",
        )

        await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="doc3",
            document_name="Markdown 1",
            document_type=DocumentType.MARKDOWN,
            content="# Content",
        )

        # Get statistics with breakdown
        stats = await memory_library.get_statistics(owner_id, include_breakdown=True)

        assert stats.document_type_breakdown is not None
        assert stats.document_type_breakdown[DocumentType.TEXT] == 2
        assert stats.document_type_breakdown[DocumentType.MARKDOWN] == 1

        # Check date range
        assert stats.created_date_range[0] <= stats.created_date_range[1]

    @pytest.mark.asyncio
    async def test_get_statistics_empty_owner(self, memory_library: LLMemory):
        """Test statistics for owner with no documents."""
        stats = await memory_library.get_statistics("non_existent_owner")

        assert stats.document_count == 0
        assert stats.chunk_count == 0
        assert stats.total_size_bytes == 0
        assert stats.created_date_range is None


class TestChunkManagementAPIs:
    """Test chunk management API methods."""

    @pytest.mark.asyncio
    async def test_get_document_chunks(self, memory_library: LLMemory):
        """Test retrieving document chunks."""
        # Add a document
        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc",
            document_name="Test Document",
            document_type=DocumentType.TEXT,
            content="Line 1\nLine 2\nLine 3\nLine 4\nLine 5" * 10,
        )

        doc_id = result.document.document_id

        # Get all chunks
        chunks = await memory_library.get_document_chunks(doc_id)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == doc_id for chunk in chunks)

        # Chunks should be ordered by index
        for i in range(1, len(chunks)):
            assert chunks[i].chunk_index >= chunks[i - 1].chunk_index

    @pytest.mark.asyncio
    async def test_get_document_chunks_with_pagination(self, memory_library: LLMemory):
        """Test retrieving chunks with pagination."""
        # Add document with content that creates multiple chunks
        content = "\n\n".join([f"Section {i}: " + "Content " * 50 for i in range(10)])

        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc_paginated",
            document_name="Large Document",
            document_type=DocumentType.TEXT,
            content=content,
        )

        # Get first 2 chunks
        chunks_page1 = await memory_library.get_document_chunks(
            result.document.document_id, limit=2, offset=0
        )

        assert len(chunks_page1) == 2

        # Get next 2 chunks
        chunks_page2 = await memory_library.get_document_chunks(
            result.document.document_id, limit=2, offset=2
        )

        assert len(chunks_page2) <= 2

        # Ensure different chunks
        page1_ids = {str(c.chunk_id) for c in chunks_page1}
        page2_ids = {str(c.chunk_id) for c in chunks_page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_get_chunk_count(self, memory_library: LLMemory):
        """Test getting chunk count for a document."""
        # Add a document with more content to ensure multiple chunks
        content = "This is a test document with substantial content. " * 50

        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc",
            document_name="Test Document",
            document_type=DocumentType.TEXT,
            content=content,
        )

        # Get chunk count
        count = await memory_library.get_chunk_count(result.document.document_id)

        assert isinstance(count, int)
        assert count > 0
        # The count should match what was reported at creation time
        # Allow for some flexibility due to potential deduplication
        assert count <= result.chunks_created

    @pytest.mark.asyncio
    async def test_chunk_apis_document_not_found(self, memory_library: LLMemory):
        """Test chunk APIs with non-existent document."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(DocumentNotFoundError):
            await memory_library.get_document_chunks(fake_id)

        with pytest.raises(DocumentNotFoundError):
            await memory_library.get_chunk_count(fake_id)


class TestBatchOperations:
    """Test batch operation API methods."""

    @pytest.mark.asyncio
    async def test_delete_documents_by_ids(self, memory_library: LLMemory):
        """Test deleting multiple documents by IDs."""
        owner_id = "test_batch_delete"
        doc_ids = []

        # Add documents
        for i in range(5):
            result = await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=DocumentType.TEXT,
                content=f"Content {i}",
            )
            doc_ids.append(result.document.document_id)

        # Delete first 3 documents
        delete_result = await memory_library.delete_documents(
            owner_id=owner_id, document_ids=doc_ids[:3]
        )

        assert isinstance(delete_result, DeleteResult)
        assert delete_result.deleted_count == 3
        assert len(delete_result.deleted_document_ids) == 3
        assert set(delete_result.deleted_document_ids) == set(doc_ids[:3])

        # Verify remaining documents
        remaining = await memory_library.list_documents(owner_id)
        assert remaining.total == 2

    @pytest.mark.asyncio
    async def test_delete_documents_by_metadata(self, memory_library: LLMemory):
        """Test deleting documents by metadata filter."""
        owner_id = "test_batch_delete_metadata"

        # Add documents with different metadata
        for i in range(6):
            category = "delete_me" if i < 4 else "keep_me"
            await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=DocumentType.TEXT,
                content=f"Content {i}",
                metadata={"category": category, "index": i},
            )

        # Delete by metadata filter
        delete_result = await memory_library.delete_documents(
            owner_id=owner_id, metadata_filter={"category": "delete_me"}
        )

        assert delete_result.deleted_count == 4

        # Verify remaining documents
        remaining = await memory_library.list_documents(owner_id)
        assert remaining.total == 2
        assert all(doc.metadata["category"] == "keep_me" for doc in remaining.documents)

    @pytest.mark.asyncio
    async def test_delete_documents_wrong_owner(self, memory_library: LLMemory):
        """Test that documents can only be deleted by their owner."""
        # Add document as one owner
        result = await memory_library.add_document(
            owner_id="owner1",
            id_at_origin="doc1",
            document_name="Document 1",
            document_type=DocumentType.TEXT,
            content="Content",
        )

        # Try to delete as different owner
        delete_result = await memory_library.delete_documents(
            owner_id="owner2", document_ids=[result.document.document_id]
        )

        assert delete_result.deleted_count == 0
        assert len(delete_result.deleted_document_ids) == 0

        # Verify document still exists
        doc = await memory_library.get_document(result.document.document_id)
        assert doc.document.owner_id == "owner1"

    @pytest.mark.asyncio
    async def test_delete_documents_no_filter_error(self, memory_library: LLMemory):
        """Test that delete requires either IDs or metadata filter."""
        with pytest.raises(ValueError) as exc_info:
            await memory_library.delete_documents("test_owner")

        assert "Either document_ids or metadata_filter must be provided" in str(exc_info.value)


class TestDocumentAddResult:
    """Test the enhanced add_document return value."""

    @pytest.mark.asyncio
    async def test_add_document_returns_statistics(self, memory_library: LLMemory):
        """Test that add_document returns proper statistics."""
        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc",
            document_name="Test Document",
            document_type=DocumentType.TEXT,
            content="This is test content that will be chunked and processed.",
            generate_embeddings=True,
        )

        # Check return type and fields
        from llmemory import DocumentAddResult

        assert isinstance(result, DocumentAddResult)
        assert isinstance(result.document, Document)
        assert result.chunks_created > 0
        assert result.embeddings_created >= 0  # May be 0 if async
        assert result.processing_time_ms > 0

        # Verify document was created
        assert result.document.document_name == "Test Document"
        assert result.document.owner_id == "test_owner"

    @pytest.mark.asyncio
    async def test_add_document_without_embeddings(self, memory_library: LLMemory):
        """Test add_document without generating embeddings."""
        result = await memory_library.add_document(
            owner_id="test_owner",
            id_at_origin="test_doc_no_embed",
            document_name="No Embeddings Doc",
            document_type=DocumentType.TEXT,
            content="Content without embeddings",
            generate_embeddings=False,
        )

        assert result.chunks_created > 0
        assert result.embeddings_created == 0
