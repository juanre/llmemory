"""Integration tests for new aword-memory API methods."""

import asyncio

import pytest
from llmemory import LLMemory, DocumentNotFoundError, DocumentType, SearchType, ValidationError


class TestAPIIntegration:
    """Test integration between different API methods."""

    @pytest.mark.asyncio
    async def test_full_document_lifecycle(self, memory_library: LLMemory):
        """Test complete document lifecycle using all APIs."""
        owner_id = "test_lifecycle"

        # 1. Add a document
        add_result = await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="lifecycle_doc",
            document_name="Lifecycle Test Document",
            document_type=DocumentType.MARKDOWN,
            content="""
            # Main Title
            This is the introduction section with important information.

            ## Section 1
            First section content goes here.

            ## Section 2
            Second section with more details.
            """,
            metadata={"status": "draft", "version": "1.0"},
        )

        doc_id = add_result.document.document_id

        # 2. List documents to verify it exists
        list_result = await memory_library.list_documents(owner_id)
        assert list_result.total == 1
        assert list_result.documents[0].document_id == doc_id

        # 3. Get the document with chunks
        doc_with_chunks = await memory_library.get_document(doc_id, include_chunks=True)
        # Allow for deduplication - actual chunks may be less than initially created
        assert doc_with_chunks.chunk_count <= add_result.chunks_created
        assert doc_with_chunks.chunk_count > 0
        assert len(doc_with_chunks.chunks) == doc_with_chunks.chunk_count

        # 4. Search for content
        search_results = await memory_library.search_with_documents(
            owner_id=owner_id,
            query_text="introduction section",
            search_type=SearchType.TEXT,
        )
        assert len(search_results.results) > 0
        assert search_results.results[0].document_name == "Lifecycle Test Document"

        # 5. Get statistics
        stats = await memory_library.get_statistics(owner_id)
        assert stats.document_count == 1
        assert stats.chunk_count <= add_result.chunks_created
        assert stats.chunk_count > 0

        # 6. Delete the document
        delete_result = await memory_library.delete_documents(
            owner_id=owner_id, document_ids=[doc_id]
        )
        assert delete_result.deleted_count == 1

        # 7. Verify deletion
        with pytest.raises(DocumentNotFoundError):
            await memory_library.get_document(doc_id)

        final_stats = await memory_library.get_statistics(owner_id)
        assert final_stats.document_count == 0

    @pytest.mark.asyncio
    async def test_multi_owner_isolation(self, memory_library: LLMemory):
        """Test that different owners' data is properly isolated."""
        owner1 = "company_a"
        owner2 = "company_b"

        # Add documents for both owners
        for owner in [owner1, owner2]:
            for i in range(3):
                await memory_library.add_document(
                    owner_id=owner,
                    id_at_origin=f"doc_{i}",
                    document_name=f"{owner} Document {i}",
                    document_type=DocumentType.TEXT,
                    content=f"Confidential content for {owner}",
                    metadata={"owner": owner},
                )

        # List documents for each owner
        owner1_docs = await memory_library.list_documents(owner1)
        owner2_docs = await memory_library.list_documents(owner2)

        assert owner1_docs.total == 3
        assert owner2_docs.total == 3

        # Ensure no cross-contamination
        assert all(doc.owner_id == owner1 for doc in owner1_docs.documents)
        assert all(doc.owner_id == owner2 for doc in owner2_docs.documents)

        # Search should respect owner boundaries
        search1 = await memory_library.search_with_documents(
            owner_id=owner1, query_text="Confidential content"
        )

        assert all(result.document_metadata.get("owner") == owner1 for result in search1.results)

        # Statistics should be owner-specific
        stats1 = await memory_library.get_statistics(owner1)
        stats2 = await memory_library.get_statistics(owner2)

        assert stats1.document_count == 3
        assert stats2.document_count == 3

        # Delete owner1's documents shouldn't affect owner2
        delete_result = await memory_library.delete_documents(
            owner_id=owner1, metadata_filter={"owner": owner1}
        )

        assert delete_result.deleted_count == 3

        # Verify owner2's documents still exist
        owner2_docs_after = await memory_library.list_documents(owner2)
        assert owner2_docs_after.total == 3

    @pytest.mark.asyncio
    async def test_search_and_chunk_retrieval(self, memory_library_with_embeddings: LLMemory):
        """Test searching and then retrieving full documents with chunks."""
        memory = memory_library_with_embeddings

        # Search for Python content
        search_results = await memory.search_with_documents(
            owner_id="test_workspace", query_text="Python programming", limit=3
        )

        assert len(search_results.results) > 0

        # For each search result, get the full document with chunks
        for result in search_results.results[:2]:  # Test first 2 results
            doc_with_chunks = await memory.get_document(result.document_id, include_chunks=True)

            # Verify we can access both search result and full document data
            assert doc_with_chunks.document.document_name == result.document_name
            assert doc_with_chunks.chunks is not None

            # Find the chunk that was in the search result
            found_chunk = False
            for chunk in doc_with_chunks.chunks:
                if chunk.chunk_id == result.chunk_id:
                    found_chunk = True
                    # Content might be truncated in search results
                    assert result.content in chunk.content or chunk.content in result.content
                    break

            assert found_chunk, "Search result chunk should be in document chunks"

    @pytest.mark.asyncio
    async def test_metadata_filtering_across_apis(self, memory_library: LLMemory):
        """Test metadata filtering works consistently across different APIs."""
        owner_id = "test_metadata_consistency"

        # Add documents with structured metadata
        projects = ["alpha", "beta", "gamma"]
        priorities = ["high", "medium", "low"]

        doc_count = 0
        for project in projects:
            for priority in priorities:
                await memory_library.add_document(
                    owner_id=owner_id,
                    id_at_origin=f"doc_{doc_count}",
                    document_name=f"Project {project} - Priority {priority}",
                    document_type=DocumentType.TEXT,
                    content=f"Work items for project {project} with {priority} priority",
                    metadata={"project": project, "priority": priority, "year": 2024},
                )
                doc_count += 1

        # Test listing with metadata filter
        high_priority_docs = await memory_library.list_documents(
            owner_id=owner_id, metadata_filter={"priority": "high"}
        )
        assert high_priority_docs.total == 3

        # Test complex metadata filter
        alpha_high = await memory_library.list_documents(
            owner_id=owner_id, metadata_filter={"project": "alpha", "priority": "high"}
        )
        assert alpha_high.total == 1
        assert alpha_high.documents[0].metadata["project"] == "alpha"

        # Test search with metadata filter
        search_results = await memory_library.search_with_documents(
            owner_id=owner_id,
            query_text="work items",
            metadata_filter={"project": "beta"},
        )

        # All results should be from beta project
        assert all(
            "beta" in result.content.lower() or result.document_metadata.get("project") == "beta"
            for result in search_results.results
        )

        # Test batch delete with metadata filter
        delete_result = await memory_library.delete_documents(
            owner_id=owner_id, metadata_filter={"priority": "low", "year": 2024}
        )
        assert delete_result.deleted_count == 3

        # Verify only low priority docs were deleted
        remaining = await memory_library.list_documents(owner_id)
        assert remaining.total == 6
        assert all(doc.metadata["priority"] != "low" for doc in remaining.documents)

    @pytest.mark.asyncio
    async def test_pagination_consistency(self, memory_library: LLMemory):
        """Test that pagination works consistently across list and chunk APIs."""
        owner_id = "test_pagination"

        # Add a document with many chunks
        large_content = "\n\n".join([f"Chapter {i}: " + "Content " * 100 for i in range(20)])

        add_result = await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="large_doc",
            document_name="Large Document",
            document_type=DocumentType.TEXT,
            content=large_content,
        )

        doc_id = add_result.document.document_id

        # Test chunk pagination
        all_chunks = []
        offset = 0
        limit = 5

        while True:
            page_chunks = await memory_library.get_document_chunks(
                doc_id, limit=limit, offset=offset
            )

            if not page_chunks:
                break

            all_chunks.extend(page_chunks)
            offset += limit

            if len(page_chunks) < limit:
                break

        # Verify we got all chunks
        total = await memory_library.get_chunk_count(doc_id)
        assert len(all_chunks) == total

        # Verify no duplicates
        chunk_ids = [str(c.chunk_id) for c in all_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

        # Verify ordering is maintained
        for i in range(1, len(all_chunks)):
            assert all_chunks[i].chunk_index >= all_chunks[i - 1].chunk_index


class TestErrorHandling:
    """Test error handling across new APIs."""

    @pytest.mark.asyncio
    async def test_invalid_owner_id(self, memory_library: LLMemory):
        """Test validation of owner_id parameter."""
        # Empty owner_id
        with pytest.raises(ValidationError) as exc_info:
            await memory_library.list_documents("")
        assert exc_info.value.field == "owner_id"

        # None owner_id
        with pytest.raises((ValidationError, TypeError)):
            await memory_library.list_documents(None)

    @pytest.mark.asyncio
    async def test_invalid_pagination_params(self, memory_library: LLMemory):
        """Test handling of invalid pagination parameters."""
        owner_id = "test_owner"

        # Add a document first
        result = await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="doc",
            document_name="Test",
            document_type=DocumentType.TEXT,
            content="Content",
        )

        # Negative limit - should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            await memory_library.list_documents(owner_id=owner_id, limit=-1)
        assert exc_info.value.field == "limit"

        # Very large offset - should return empty results
        docs = await memory_library.list_documents(owner_id=owner_id, offset=1000000)
        assert len(docs.documents) == 0
        assert docs.total == 1  # Still shows correct total

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_library: LLMemory):
        """Test concurrent API operations don't interfere with each other."""
        owner_id = "test_concurrent"

        # Add initial documents
        doc_ids = []
        for i in range(5):
            result = await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=DocumentType.TEXT,
                content=f"Content for document {i}",
            )
            doc_ids.append(result.document.document_id)

        # Run multiple operations concurrently
        async def list_docs():
            return await memory_library.list_documents(owner_id)

        async def get_doc(doc_id):
            return await memory_library.get_document(doc_id)

        async def search_docs():
            return await memory_library.search_with_documents(
                owner_id=owner_id, query_text="Content"
            )

        async def get_stats():
            return await memory_library.get_statistics(owner_id)

        # Execute concurrently
        results = await asyncio.gather(
            list_docs(),
            get_doc(doc_ids[0]),
            get_doc(doc_ids[1]),
            search_docs(),
            get_stats(),
            return_exceptions=True,
        )

        # Verify all operations succeeded
        list_result, doc1, doc2, search_result, stats = results

        assert not isinstance(list_result, Exception)
        assert list_result.total == 5

        assert not isinstance(doc1, Exception)
        assert doc1.document.document_id == doc_ids[0]

        assert not isinstance(search_result, Exception)
        assert len(search_result.results) > 0

        assert not isinstance(stats, Exception)
        assert stats.document_count == 5


class TestPerformanceConsiderations:
    """Test performance aspects of new APIs."""

    @pytest.mark.asyncio
    async def test_large_document_handling(self, memory_library: LLMemory):
        """Test handling of large documents."""
        owner_id = "test_performance"

        # Create a large document (1MB+)
        large_content = "Large content block. " * 50000

        start_time = asyncio.get_event_loop().time()

        add_result = await memory_library.add_document(
            owner_id=owner_id,
            id_at_origin="large_doc",
            document_name="Large Document",
            document_type=DocumentType.TEXT,
            content=large_content,
            generate_embeddings=False,  # Skip embeddings for performance test
        )

        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        # Should complete in reasonable time (< 10 seconds)
        assert processing_time < 10.0
        assert add_result.chunks_created > 0

        # Test retrieving large document
        doc_with_chunks = await memory_library.get_document(
            add_result.document.document_id, include_chunks=True
        )

        assert doc_with_chunks.chunk_count <= add_result.chunks_created
        assert doc_with_chunks.chunk_count > 0

        # Clean up
        await memory_library.delete_document(add_result.document.document_id)

    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, memory_library: LLMemory):
        """Test performance of batch operations."""
        owner_id = "test_batch_perf"

        # Add many documents
        doc_ids = []
        start_time = asyncio.get_event_loop().time()

        for i in range(50):
            result = await memory_library.add_document(
                owner_id=owner_id,
                id_at_origin=f"doc_{i}",
                document_name=f"Document {i}",
                document_type=DocumentType.TEXT,
                content=f"Content {i}",
                generate_embeddings=False,
                metadata={"batch": i // 10},  # Group into batches of 10
            )
            doc_ids.append(result.document.document_id)

        add_time = asyncio.get_event_loop().time() - start_time

        # Test listing performance
        start_time = asyncio.get_event_loop().time()
        list_result = await memory_library.list_documents(owner_id=owner_id, limit=100)
        list_time = asyncio.get_event_loop().time() - start_time

        assert list_result.total == 50
        assert list_time < 1.0  # Should be fast

        # Test batch delete performance
        start_time = asyncio.get_event_loop().time()
        delete_result = await memory_library.delete_documents(
            owner_id=owner_id, metadata_filter={"batch": 0}  # Delete first batch
        )
        delete_time = asyncio.get_event_loop().time() - start_time

        assert delete_result.deleted_count == 10
        assert delete_time < 2.0  # Should be reasonably fast
