# ABOUTME: High-level async library interface providing the main public API for llmemory document operations.
# ABOUTME: Implements document storage, search, deletion, and embedding management with comprehensive validation and error handling.

"""Main library interface for aword-memory.

This module provides the public API for integrating aword-memory into any application.
It handles async operations with clean interfaces and supports
multi-tenant configurations through owner_id filtering.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pgdbm import AsyncDatabaseManager

from .batch_processor import (BackgroundEmbeddingProcessor,
                              BatchEmbeddingProcessor)
from .config import LLMemoryConfig, get_config
from .embedding_providers import EmbeddingProvider, EmbeddingProviderFactory
from .embeddings import EmbeddingGenerator
from .exceptions import (ConfigurationError, DocumentNotFoundError,
                         ValidationError)
from .manager import MemoryManager
from .models import (ChunkingConfig, DeleteResult, Document, DocumentAddResult,
                     DocumentChunk, DocumentListResult, DocumentType,
                     DocumentWithChunks, EnrichedSearchResult, OwnerStatistics,
                     SearchQuery, SearchResult, SearchResultWithDocuments,
                     SearchType)
from .search_optimizer import OptimizedAsyncSearch
from .validators import get_validator

logger = logging.getLogger(__name__)


class LLMemory:
    """Main library interface for llmemory.

    This class provides a clean API for document ingestion, search, and management
    with support for asynchronous operations.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        config: Optional[LLMemoryConfig] = None,
        db_manager: Optional[AsyncDatabaseManager] = None,
    ):
        """
        Initialize the LLMemory library.

        Args:
            connection_string: PostgreSQL connection string (ignored if db_manager provided)
            openai_api_key: OpenAI API key for embeddings (optional, can be set later)
            config: Optional configuration object (defaults to config from environment)
            db_manager: Optional AsyncDatabaseManager for shared pool integration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not connection_string and not db_manager:
            raise ConfigurationError(
                "Either connection string or db_manager is required"
            )

        self.connection_string = connection_string
        self.openai_api_key = openai_api_key
        self.config = config or get_config()
        self.validator = get_validator()
        self._external_db = db_manager is not None
        self._db_manager = db_manager

        # Update OpenAI API key in config if provided
        if self.openai_api_key and "openai" in self.config.embedding.providers:
            self.config.embedding.providers["openai"].api_key = self.openai_api_key

        self._manager: Optional[MemoryManager] = None
        self._embedding_providers: Dict[str, EmbeddingProvider] = {}
        self._batch_processor: Optional[BatchEmbeddingProcessor] = None
        self._background_processor: Optional[BackgroundEmbeddingProcessor] = None
        self._optimized_search: Optional[OptimizedAsyncSearch] = None
        self._initialized = False

    @classmethod
    def from_db_manager(
        cls,
        db_manager: AsyncDatabaseManager,
        openai_api_key: Optional[str] = None,
        config: Optional[LLMemoryConfig] = None,
    ) -> "LLMemory":
        """
        Create LLMemory instance from existing AsyncDatabaseManager.

        This is the recommended way to use llmemory when integrating
        with other services that share a database connection pool.

        Args:
            db_manager: Existing AsyncDatabaseManager instance (e.g., from shared pool)
                        Should already have the correct schema set
            openai_api_key: OpenAI API key for embeddings
            config: Optional configuration object

        Returns:
            LLMemory instance configured for external db management
        """
        instance = cls(
            connection_string=None,
            openai_api_key=openai_api_key,
            config=config,
            db_manager=db_manager,
        )
        return instance

    # Initialization Methods

    async def initialize(self) -> None:
        """Initialize the library for operations."""
        if self._initialized:
            return

        # Create and initialize the manager
        if self._external_db:
            # Use external db manager
            self._manager = MemoryManager.from_db_manager(self._db_manager)
            await self._manager.initialize()
        else:
            # Create own db manager
            self._manager = await MemoryManager.create(
                self.connection_string,
                schema=self.config.database.schema_name,
                enable_monitoring=self.config.enable_metrics,
                min_connections=self.config.database.min_pool_size,
                max_connections=self.config.database.max_pool_size,
            )

        # Initialize embedding providers registry
        await self._ensure_providers_registered()

        # Initialize optimized search
        self._optimized_search = OptimizedAsyncSearch(
            db=self._manager.db.db_manager,
            cache_ttl=300,
            max_concurrent_queries=100,
            enable_query_optimization=True,
        )

        self._initialized = True
        logger.info("LLMemory initialized successfully")

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self._background_processor:
            await self._background_processor.stop()
        # Note: BatchEmbeddingProcessor has no close() method - no resources to clean up
        if self._manager:
            await self._manager.close()

        self._initialized = False
        logger.info("LLMemory closed")

    # Document Management

    async def add_document(
        self,
        owner_id: str,
        id_at_origin: str,
        document_name: str,
        document_type: Union[DocumentType, str],
        content: str,
        document_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "hierarchical",
        chunking_config: Optional[ChunkingConfig] = None,
        generate_embeddings: bool = True,
    ) -> DocumentAddResult:
        """
        Add a document and process it into chunks.

        Args:
            owner_id: Owner identifier for multi-tenancy
            id_at_origin: Origin identifier within owner
            document_name: Name of the document
            document_type: Type of document
            content: Full document content
            document_date: Optional document date
            metadata: Optional metadata
            chunking_strategy: Strategy to use for chunking
            chunking_config: Optional chunking configuration
            generate_embeddings: Whether to generate embeddings immediately

        Returns:
            DocumentAddResult with document and processing statistics

        Raises:
            ValidationError: If input validation fails
            DatabaseError: If database operation fails
        """
        await self._ensure_initialized()

        start_time = time.time()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)
        self.validator.validate_id_at_origin(id_at_origin)
        self.validator.validate_document_name(document_name)

        if isinstance(document_type, str):
            document_type = DocumentType(document_type)

        # Process the document
        doc, chunks = await self._manager.process_document(
            owner_id=owner_id,
            id_at_origin=id_at_origin,
            document_name=document_name,
            document_type=document_type,
            content=content,
            document_date=document_date,
            metadata=metadata,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
        )

        embeddings_created = 0
        # Generate embeddings if requested
        if generate_embeddings and chunks:
            embeddings_created = await self._generate_embeddings_for_chunks(chunks)

        processing_time_ms = (time.time() - start_time) * 1000

        return DocumentAddResult(
            document=doc,
            chunks_created=len(chunks),
            embeddings_created=embeddings_created,
            processing_time_ms=processing_time_ms,
        )

    async def search(
        self,
        owner_id: str,
        query_text: str,
        search_type: Union[SearchType, str] = SearchType.HYBRID,
        limit: int = 10,
        id_at_origin: Optional[str] = None,
        id_at_origins: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        include_parent_context: bool = False,
        context_window: int = 2,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Search for documents.

        Args:
            owner_id: Owner identifier for filtering
            query_text: Search query text
            search_type: Type of search to perform
            limit: Maximum number of results
            id_at_origin: Single origin ID filter
            id_at_origins: Multiple origin IDs filter
            metadata_filter: JSONB metadata filter
            date_from: Start date filter
            date_to: End date filter
            include_parent_context: Whether to include parent chunks
            context_window: Number of parent chunks to include
            alpha: Weight for hybrid search (0=text only, 1=vector only)

        Returns:
            List of SearchResult instances

        Raises:
            ValidationError: If input validation fails
            SearchError: If search operation fails
        """
        await self._ensure_initialized()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)
        self.validator.validate_query_text(query_text)

        if isinstance(search_type, str):
            search_type = SearchType(search_type)

        # Create search query
        search_query = SearchQuery(
            owner_id=owner_id,
            query_text=query_text,
            search_type=search_type,
            limit=limit,
            id_at_origin=id_at_origin,
            id_at_origins=id_at_origins,
            metadata_filter=metadata_filter,
            date_from=date_from,
            date_to=date_to,
            include_parent_context=include_parent_context,
            context_window=context_window,
            alpha=alpha,
        )

        # Generate query embedding if needed
        query_embedding = None
        if search_type in [SearchType.VECTOR, SearchType.HYBRID]:
            query_embedding = await self._generate_query_embedding(query_text)

        # Use manager search (optimized search can be enabled via search_optimizer module)
        return await self._manager.search(search_query, query_embedding)

    async def delete_document(self, document_id: Union[UUID, str]) -> None:
        """
        Delete a document and all its chunks.

        Args:
            document_id: Document ID to delete

        Raises:
            ResourceNotFoundError: If document not found
            DatabaseError: If deletion fails
        """
        await self._ensure_initialized()

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        await self._manager.delete_document(document_id)

    # New API Methods

    async def list_documents(
        self,
        owner_id: str,
        limit: int = 20,
        offset: int = 0,
        document_type: Optional[DocumentType] = None,
        order_by: Literal["created_at", "updated_at", "document_name"] = "created_at",
        order_desc: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> DocumentListResult:
        """
        List documents for an owner with pagination and filtering.

        Args:
            owner_id: Owner identifier
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            document_type: Filter by document type
            order_by: Field to sort by
            order_desc: Sort in descending order
            metadata_filter: Filter by metadata fields (e.g., {"category": "reports"})

        Returns:
            DocumentListResult with:
                - documents: List[Document] - Document objects with metadata
                - total: int - Total count of matching documents
                - limit: int - Applied limit
                - offset: int - Applied offset
        """
        await self._ensure_initialized()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)

        # Validate pagination parameters
        if limit < 0:
            raise ValidationError("limit", "Limit must be non-negative")
        if offset < 0:
            raise ValidationError("offset", "Offset must be non-negative")

        # Build query
        query_parts = [
            """
        SELECT
            document_id, owner_id, id_at_origin, document_type, document_name,
            document_date, metadata, created_at, updated_at
        FROM {{tables.documents}}
        WHERE owner_id = $1
        """
        ]

        params = [owner_id]
        param_count = 2

        # Add optional filters
        if document_type:
            query_parts.append(f"AND document_type = ${param_count}")
            params.append(document_type.value)
            param_count += 1

        if metadata_filter:
            query_parts.append(f"AND metadata @> ${param_count}::jsonb")
            params.append(json.dumps(metadata_filter))
            param_count += 1

        # Add ordering with whitelist to avoid SQL injection via identifiers
        allowed_order_columns = {"created_at", "updated_at", "document_name"}
        if order_by not in allowed_order_columns:
            raise ValidationError("order_by", "Invalid order_by column")
        order_direction = "DESC" if order_desc else "ASC"
        query_parts.append(f"ORDER BY {order_by} {order_direction}")

        # Add pagination
        query_parts.append(f"LIMIT ${param_count} OFFSET ${param_count + 1}")
        params.extend([limit, offset])

        query = "\n".join(query_parts)

        # Get documents
        rows = await self._manager.db.db.fetch_all(query, *params)

        # Get total count
        count_query_parts = [
            """
        SELECT COUNT(*) as count
        FROM {{tables.documents}}
        WHERE owner_id = $1
        """
        ]

        count_params = [owner_id]

        if document_type:
            count_query_parts.append("AND document_type = $2")
            count_params.append(document_type.value)

        if metadata_filter:
            count_query_parts.append(f"AND metadata @> ${len(count_params) + 1}::jsonb")
            count_params.append(json.dumps(metadata_filter))

        count_query = "\n".join(count_query_parts)
        count_result = await self._manager.db.db.fetch_one(count_query, *count_params)
        total = count_result["count"] if count_result else 0

        # Convert to Document objects
        documents = []
        for row in rows:
            # Parse metadata if it's a string
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            doc = Document(
                document_id=UUID(str(row["document_id"])),
                owner_id=row["owner_id"],
                id_at_origin=row["id_at_origin"],
                document_type=DocumentType(row["document_type"]),
                document_name=row["document_name"],
                document_date=row["document_date"],
                metadata=metadata,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            documents.append(doc)

        return DocumentListResult(
            documents=documents, total=total, limit=limit, offset=offset
        )

    async def get_document(
        self,
        document_id: Union[str, UUID],
        include_chunks: bool = False,
        include_embeddings: bool = False,
    ) -> DocumentWithChunks:
        """
        Retrieve a specific document with optional chunks.

        Args:
            document_id: Document identifier
            include_chunks: Include all chunks for this document
            include_embeddings: Include embeddings with chunks (if include_chunks=True)

        Returns:
            DocumentWithChunks with:
                - document: Document object
                - chunks: Optional[List[DocumentChunk]] - Chunks if requested
                - chunk_count: int - Total number of chunks

        Raises:
            DocumentNotFoundError: If document doesn't exist
            PermissionError: If caller doesn't have access
        """
        await self._ensure_initialized()

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        # Get document from database
        doc_row = await self._manager.db.get_document(str(document_id))

        if not doc_row:
            raise DocumentNotFoundError(document_id)

        # Convert to Document object
        # Parse metadata if it's a string
        metadata = doc_row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        document = Document(
            document_id=UUID(str(doc_row["document_id"])),
            owner_id=doc_row["owner_id"],
            id_at_origin=doc_row["id_at_origin"],
            document_type=DocumentType(doc_row["document_type"]),
            document_name=doc_row["document_name"],
            document_date=doc_row["document_date"],
            metadata=metadata,
            created_at=doc_row["created_at"],
            updated_at=doc_row["updated_at"],
        )

        # Get chunk count
        chunk_count = await self._manager.db.count_chunks(str(document_id))

        # Get chunks if requested
        chunks = None
        if include_chunks:
            chunk_rows = await self._manager.db.get_document_chunks(
                str(document_id), include_embeddings=include_embeddings
            )

            chunks = []
            for row in chunk_rows:
                # Parse metadata if it's a string
                chunk_metadata = row["metadata"]
                if isinstance(chunk_metadata, str):
                    chunk_metadata = json.loads(chunk_metadata)

                chunk = DocumentChunk(
                    chunk_id=UUID(str(row["chunk_id"])),
                    document_id=UUID(str(row["document_id"])),
                    parent_chunk_id=(
                        UUID(str(row["parent_chunk_id"]))
                        if row["parent_chunk_id"]
                        else None
                    ),
                    chunk_index=row["chunk_index"],
                    chunk_level=row["chunk_level"],
                    content=row["content"],
                    content_hash=row["content_hash"],
                    token_count=row["token_count"],
                    metadata=chunk_metadata,
                    created_at=row["created_at"],
                )
                chunks.append(chunk)

        return DocumentWithChunks(
            document=document, chunks=chunks, chunk_count=chunk_count
        )

    async def search_with_documents(
        self,
        owner_id: str,
        query_text: str,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_document_metadata: bool = True,
    ) -> SearchResultWithDocuments:
        """
        Search with enriched results including document metadata.

        Args:
            owner_id: Owner identifier
            query_text: Search query
            search_type: Type of search (VECTOR, TEXT, HYBRID)
            limit: Maximum results
            metadata_filter: Filter by chunk or document metadata
            include_document_metadata: Include document-level metadata

        Returns:
            SearchResultWithDocuments with:
                - results: List[EnrichedSearchResult] where each result has:
                    - chunk_id, content, score (existing fields)
                    - document_name, document_type, document_metadata (new fields)
                - total: int - Total matching results
        """
        await self._ensure_initialized()

        # Perform the regular search
        search_results = await self.search(
            owner_id=owner_id,
            query_text=query_text,
            search_type=search_type,
            limit=limit,
            metadata_filter=metadata_filter,
        )

        # If we need document metadata, enrich the results
        enriched_results = []
        if include_document_metadata and search_results:
            # Get unique document IDs
            doc_ids = list(set(str(result.document_id) for result in search_results))

            # Fetch document metadata in batch - use the manager's prepared query
            doc_query = self._manager.db.db._prepare_query(
                """
            SELECT document_id, document_name, document_type, metadata
            FROM {{tables.documents}}
            WHERE document_id = ANY($1::uuid[])
              AND owner_id = $2
            """
            )
            doc_rows = await self._manager.db.db.fetch_all(doc_query, doc_ids, owner_id)

            # Create lookup map (convert UUID to string for consistent lookup)
            doc_map = {str(row["document_id"]): row for row in doc_rows}

            # Enrich results
            for result in search_results:
                doc_data = doc_map.get(str(result.document_id), {})

                # Parse document metadata if it's a string
                doc_metadata = doc_data.get("metadata", {})
                if isinstance(doc_metadata, str):
                    doc_metadata = json.loads(doc_metadata)

                enriched_result = EnrichedSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    metadata=result.metadata,
                    score=result.score,
                    similarity=result.similarity,
                    text_rank=result.text_rank,
                    rrf_score=result.rrf_score,
                    parent_chunks=result.parent_chunks,
                    document_name=doc_data.get("document_name", ""),
                    document_type=doc_data.get("document_type", ""),
                    document_metadata=doc_metadata,
                )
                enriched_results.append(enriched_result)
        else:
            # Convert regular results to enriched results without doc metadata
            for result in search_results:
                enriched_result = EnrichedSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    metadata=result.metadata,
                    score=result.score,
                    similarity=result.similarity,
                    text_rank=result.text_rank,
                    rrf_score=result.rrf_score,
                    parent_chunks=result.parent_chunks,
                    document_name="",
                    document_type="",
                    document_metadata={},
                )
                enriched_results.append(enriched_result)

        return SearchResultWithDocuments(
            results=enriched_results, total=len(enriched_results)
        )

    async def get_statistics(
        self, owner_id: str, include_breakdown: bool = False
    ) -> OwnerStatistics:
        """
        Get statistics for an owner's documents.

        Args:
            owner_id: Owner identifier
            include_breakdown: Include breakdown by document type

        Returns:
            OwnerStatistics with:
                - document_count: int
                - chunk_count: int
                - total_size_bytes: int (estimated)
                - document_type_breakdown: Optional[Dict[DocumentType, int]]
                - created_date_range: Tuple[datetime, datetime]
        """
        await self._ensure_initialized()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)

        # Get document count
        doc_count = await self._manager.db.count_documents(owner_id)

        # Get chunk count and total size estimate
        chunk_query = """
        SELECT
            COUNT(*) as chunk_count,
            SUM(token_count * 4) as estimated_bytes
        FROM {{tables.document_chunks}} c
        JOIN {{tables.documents}} d ON c.document_id = d.document_id
        WHERE d.owner_id = $1
        """
        chunk_result = await self._manager.db.db.fetch_one(chunk_query, owner_id)

        chunk_count = chunk_result["chunk_count"] if chunk_result else 0
        total_size_bytes = (
            int(chunk_result["estimated_bytes"] or 0) if chunk_result else 0
        )

        # Get document type breakdown if requested
        type_breakdown = None
        if include_breakdown:
            breakdown_query = """
            SELECT document_type, COUNT(*) as count
            FROM {{tables.documents}}
            WHERE owner_id = $1
            GROUP BY document_type
            """
            breakdown_rows = await self._manager.db.db.fetch_all(
                breakdown_query, owner_id
            )

            type_breakdown = {
                DocumentType(row["document_type"]): row["count"]
                for row in breakdown_rows
            }

        # Get date range
        date_range_query = """
        SELECT
            MIN(created_at) as min_date,
            MAX(created_at) as max_date
        FROM {{tables.documents}}
        WHERE owner_id = $1
        """
        date_result = await self._manager.db.db.fetch_one(date_range_query, owner_id)

        created_date_range = None
        if date_result and date_result["min_date"] and date_result["max_date"]:
            created_date_range = (date_result["min_date"], date_result["max_date"])

        return OwnerStatistics(
            document_count=doc_count,
            chunk_count=chunk_count,
            total_size_bytes=total_size_bytes,
            document_type_breakdown=type_breakdown,
            created_date_range=created_date_range,
        )

    async def get_document_chunks(
        self,
        document_id: Union[str, UUID],
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[DocumentChunk]:
        """
        Get chunks for a specific document with pagination.

        Args:
            document_id: Document identifier
            limit: Maximum number of chunks to return
            offset: Number of chunks to skip

        Returns:
            List of DocumentChunk objects

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        await self._ensure_initialized()

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        # Validate pagination parameters
        if limit is not None and limit < 0:
            raise ValidationError("limit", "Limit must be non-negative")
        if offset < 0:
            raise ValidationError("offset", "Offset must be non-negative")

        # Check if document exists
        if not await self._manager.db.document_exists(str(document_id)):
            raise DocumentNotFoundError(document_id)

        # Build query - use prepared query for schema support
        query = self._manager.db.db._prepare_query(
            """
        SELECT
            chunk_id, document_id, parent_chunk_id, chunk_index,
            chunk_level, content, content_hash, token_count,
            metadata, created_at
        FROM {{tables.document_chunks}}
        WHERE document_id = $1
        ORDER BY chunk_index, chunk_id
        """
        )

        params = [str(document_id)]

        if limit is not None:
            query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
            params.extend([limit, offset])

        rows = await self._manager.db.db.fetch_all(query, *params)

        chunks = []
        for row in rows:
            # Parse metadata if it's a string
            chunk_metadata = row["metadata"]
            if isinstance(chunk_metadata, str):
                chunk_metadata = json.loads(chunk_metadata)

            chunk = DocumentChunk(
                chunk_id=UUID(str(row["chunk_id"])),
                document_id=UUID(str(row["document_id"])),
                parent_chunk_id=(
                    UUID(str(row["parent_chunk_id"]))
                    if row["parent_chunk_id"]
                    else None
                ),
                chunk_index=row["chunk_index"],
                chunk_level=row["chunk_level"],
                content=row["content"],
                content_hash=row["content_hash"],
                token_count=row["token_count"],
                metadata=chunk_metadata,
                created_at=row["created_at"],
            )
            chunks.append(chunk)

        return chunks

    async def get_chunk_count(self, document_id: Union[str, UUID]) -> int:
        """
        Get the number of chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of chunks

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        await self._ensure_initialized()

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        # Check if document exists
        if not await self._manager.db.document_exists(str(document_id)):
            raise DocumentNotFoundError(document_id)

        return await self._manager.db.count_chunks(str(document_id))

    async def delete_documents(
        self,
        owner_id: str,
        document_ids: Optional[List[Union[str, UUID]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> DeleteResult:
        """
        Delete multiple documents.

        Args:
            owner_id: Owner identifier
            document_ids: Specific documents to delete
            metadata_filter: Delete all documents matching filter

        Returns:
            DeleteResult with:
                - deleted_count: int
                - deleted_document_ids: List[UUID]

        Raises:
            ValueError: If neither document_ids nor metadata_filter provided
        """
        await self._ensure_initialized()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)

        if not document_ids and not metadata_filter:
            raise ValueError("Either document_ids or metadata_filter must be provided")

        deleted_ids = []

        if document_ids:
            # Delete specific documents
            for doc_id in document_ids:
                if isinstance(doc_id, str):
                    doc_id = UUID(doc_id)

                # Check if document exists and belongs to owner
                doc_row = await self._manager.db.get_document(str(doc_id))
                if doc_row and doc_row["owner_id"] == owner_id:
                    await self._manager.delete_document(doc_id)
                    deleted_ids.append(doc_id)
                elif doc_row:
                    logger.warning(
                        f"Document {doc_id} exists but belongs to different owner, skipping"
                    )
                else:
                    logger.debug(f"Document {doc_id} not found, skipping")

        elif metadata_filter:
            # Find documents matching filter
            query = """
            SELECT document_id
            FROM {{tables.documents}}
            WHERE owner_id = $1 AND metadata @> $2::jsonb
            """
            rows = await self._manager.db.db.fetch_all(
                query, owner_id, json.dumps(metadata_filter)
            )

            # Delete each matching document
            for row in rows:
                doc_id = UUID(str(row["document_id"]))
                await self._manager.delete_document(doc_id)
                deleted_ids.append(doc_id)

        return DeleteResult(
            deleted_count=len(deleted_ids), deleted_document_ids=deleted_ids
        )

    # Embedding Management

    async def process_pending_embeddings(
        self, batch_size: int = 100, max_batches: int = 10
    ) -> int:
        """
        Process pending embeddings in batches.

        Args:
            batch_size: Number of embeddings per batch
            max_batches: Maximum number of batches to process

        Returns:
            Number of embeddings processed

        Raises:
            EmbeddingError: If embedding generation fails
        """
        await self._ensure_initialized()

        if not self._batch_processor:
            self._batch_processor = BatchEmbeddingProcessor(
                embedding_generator=await self._get_embedding_generator(),
                batch_size=batch_size,
            )

        return await self._batch_processor.process_pending_embeddings(max_batches)

    async def start_background_processing(
        self, interval_seconds: int = 60, batch_size: int = 100
    ) -> None:
        """
        Start background embedding processing.

        Args:
            interval_seconds: Seconds between processing runs
            batch_size: Number of embeddings per batch
        """
        await self._ensure_initialized()

        if self._background_processor:
            await self._background_processor.stop()

        self._background_processor = BackgroundEmbeddingProcessor(
            self._manager,
            await self._get_embedding_generator(),
            interval_seconds=interval_seconds,
            batch_size=batch_size,
        )

        await self._background_processor.start()

    # Private Methods

    async def _ensure_initialized(self) -> None:
        """Ensure the library is initialized."""
        if not self._initialized:
            await self.initialize()

    async def _ensure_providers_registered(self) -> None:
        """Ensure embedding providers are registered in database."""
        # This would typically register providers in the embedding_providers table
        # For now, we'll skip this as it's handled by migrations
        pass

    async def _get_embedding_generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator."""
        provider = await self._get_embedding_provider()
        return EmbeddingGenerator(provider)

    async def _get_embedding_provider(
        self, provider_id: Optional[str] = None
    ) -> EmbeddingProvider:
        """Get embedding provider instance."""
        if provider_id is None:
            provider_id = self.config.embedding.default_provider

        if provider_id not in self._embedding_providers:
            provider_config = self.config.embedding.providers.get(provider_id)
            if not provider_config:
                raise ConfigurationError(f"Provider {provider_id} not configured")

            self._embedding_providers[provider_id] = (
                EmbeddingProviderFactory.create_provider(provider_id, provider_config)
            )

        return self._embedding_providers[provider_id]

    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query."""
        generator = await self._get_embedding_generator()
        return await generator.generate_embedding(query_text)

    async def _generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Generate embeddings for a list of chunks. Returns count of successful embeddings."""
        generator = await self._get_embedding_generator()
        successful_count = 0

        for chunk in chunks:
            try:
                embedding = await generator.generate_embedding(chunk.content)
                await self._manager.update_chunk_embedding(chunk.chunk_id, embedding)
                successful_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}"
                )

        return successful_count

    # Context Manager Support

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @property
    def db_manager(self) -> Optional[AsyncDatabaseManager]:
        """Get the underlying database manager for health checks and monitoring.

        Returns None if not initialized.
        """
        if self._initialized and self._manager:
            return self._manager.db.db_manager
        return None
