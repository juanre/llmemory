# ABOUTME: High-level async manager coordinating document operations, embedding generation, and search across llmemory.
# ABOUTME: Orchestrates document lifecycle from ingestion through chunking, enrichment, embedding, and retrieval with validation.

"""High-level async manager for aword-memory operations."""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pgdbm import AsyncDatabaseManager

from .chunking import get_chunker
from .db import MemoryDatabase, create_memory_db_manager
from .language_processing import detect_and_process_language
from .models import (ChunkingConfig, Document, DocumentChunk, DocumentType,
                     EmbeddingJob, EmbeddingStatus, SearchQuery, SearchResult)

logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level async interface for document memory operations."""

    def __init__(self, db: Optional[MemoryDatabase] = None, external_db: bool = False):
        """
        Initialize the memory manager.

        Args:
            db: Existing MemoryDatabase instance
            external_db: Whether the db is externally managed
        """
        if db is None:
            raise ValueError(
                "MemoryDatabase instance is required. Use create_memory_manager() to create a new instance."
            )

        self.db = db
        self._external_db = external_db
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the manager and database."""
        if self._initialized:
            return

        await self.db.initialize()
        await self.db.apply_migrations()
        self._initialized = True

    @classmethod
    async def create(
        cls, connection_string: Optional[str] = None, **kwargs
    ) -> "MemoryManager":
        """
        Create and initialize a new MemoryManager instance.

        Args:
            connection_string: PostgreSQL connection string
            **kwargs: Additional arguments for create_memory_db_manager

        Returns:
            Initialized MemoryManager instance
        """
        db_manager = await create_memory_db_manager(connection_string, **kwargs)
        db = MemoryDatabase(db_manager)
        manager = cls(db)
        await manager.initialize()
        return manager

    @classmethod
    def from_db_manager(cls, db_manager: AsyncDatabaseManager) -> "MemoryManager":
        """
        Create MemoryManager instance from existing AsyncDatabaseManager.

        This is used when integrating with a shared connection pool.

        Args:
            db_manager: Existing AsyncDatabaseManager instance (e.g., from shared pool)
                        Should already have the correct schema set

        Returns:
            MemoryManager instance configured for external db management
        """
        memory_db = MemoryDatabase.from_manager(db_manager)
        return cls(memory_db, external_db=True)

    async def add_document(
        self,
        owner_id: str,
        id_at_origin: str,
        document_name: str,
        document_type: DocumentType,
        document_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Add a new document to the system.

        Args:
            owner_id: Owner identifier for filtering (e.g., workspace_id)
            id_at_origin: Origin identifier within owner (user ID, thread ID, etc.)
            document_name: Name of the document
            document_type: Type of document
            document_date: Optional document date
            metadata: Optional metadata

        Returns:
            Created Document instance
        """
        doc = Document(
            owner_id=owner_id,
            id_at_origin=id_at_origin,
            document_name=document_name,
            document_type=document_type,
            document_date=document_date,
            metadata=metadata or {},
        )

        query = """
        INSERT INTO {{tables.documents}} (
            document_id, owner_id, id_at_origin, document_type, document_name,
            document_date, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        RETURNING document_id, created_at, updated_at
        """

        result = await self.db.db_manager.fetch_one(
            query,
            str(doc.document_id),
            doc.owner_id,
            doc.id_at_origin,
            doc.document_type.value,
            doc.document_name,
            doc.document_date,
            json.dumps(doc.metadata),
        )

        doc.created_at = result["created_at"]
        doc.updated_at = result["updated_at"]

        logger.info(f"Added document {doc.document_id} ({doc.document_name})")
        return doc

    async def process_document(
        self,
        owner_id: str,
        id_at_origin: str,
        document_name: str,
        document_type: DocumentType,
        content: str,
        document_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "hierarchical",
        chunking_config: Optional[ChunkingConfig] = None,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """
        Process a complete document: add it and create chunks.

        Args:
            owner_id: Owner identifier for filtering (e.g., workspace_id)
            id_at_origin: Origin identifier within owner (user ID, thread ID, etc.)
            document_name: Name of the document
            document_type: Type of document
            content: Full document content
            document_date: Optional document date
            metadata: Optional metadata
            chunking_strategy: Strategy to use for chunking
            chunking_config: Optional chunking configuration

        Returns:
            Tuple of (Document, List[DocumentChunk])
        """
        # Detect and process language
        language_info = detect_and_process_language(content)

        # Add language info to metadata
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "language": language_info["primary_language"],
                "language_confidence": language_info["primary_confidence"],
                "is_multilingual": language_info["is_multilingual"],
                "detected_languages": language_info["languages"],
                "text_search_config": language_info["text_search_config"],
            }
        )

        # Add document
        doc = await self.add_document(
            owner_id=owner_id,
            id_at_origin=id_at_origin,
            document_name=document_name,
            document_type=document_type,
            document_date=document_date,
            metadata=metadata,
        )

        # Get chunker and create chunks based on document type
        if (
            document_type in [DocumentType.EMAIL, DocumentType.CHAT]
            and chunking_strategy == "hierarchical"
        ):
            # Use semantic chunker for email and chat
            chunker = get_chunker("semantic", chunking_config)

            if document_type == DocumentType.EMAIL:
                chunks = chunker.chunk_email(
                    text=content, document_id=str(doc.document_id)
                )
            else:  # DocumentType.CHAT
                chunks = chunker.chunk_chat(
                    text=content, document_id=str(doc.document_id)
                )

            # Add base metadata to all chunks
            for chunk in chunks:
                chunk.metadata.update(metadata or {})
        else:
            # Use specified chunker for other document types
            chunker = get_chunker(chunking_strategy, chunking_config)
            chunks = chunker.chunk_document(
                text=content,
                document_id=str(doc.document_id),
                document_type=document_type,
                base_metadata=metadata or {},
            )

        # Store chunks in database
        stored_chunks = []
        async with self.db.db_manager.transaction() as conn:
            # Group chunks by parent for efficient storage
            parent_chunks = [c for c in chunks if c.parent_chunk_id is None]

            for parent in parent_chunks:
                # Store parent chunk - pass full chunk data
                stored_parent = await self.add_chunks_with_details(
                    document_id=doc.document_id, chunks=[parent], conn=conn
                )
                if stored_parent:
                    stored_chunks.extend(stored_parent)
                    parent_id = stored_parent[0].chunk_id

                    # Store child chunks
                    child_chunks = [
                        c for c in chunks if c.parent_chunk_id == parent.chunk_id
                    ]
                    for child in child_chunks:
                        # Update parent reference to stored parent
                        child.parent_chunk_id = parent_id
                        stored_child = await self.add_chunks_with_details(
                            document_id=doc.document_id, chunks=[child], conn=conn
                        )
                        if stored_child:
                            stored_chunks.extend(stored_child)

        logger.info(
            f"Processed document {doc.document_id} into {len(stored_chunks)} chunks"
        )
        return doc, stored_chunks

    async def add_chunks(
        self,
        document_id: UUID,
        chunks: List[Tuple[str, Dict[str, Any]]],
        parent_chunk_id: Optional[UUID] = None,
        chunk_level: int = 0,
        conn: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Add multiple chunks to a document.

        Args:
            document_id: Document ID
            chunks: List of (content, metadata) tuples
            parent_chunk_id: Optional parent chunk for hierarchical chunking
            chunk_level: Level in hierarchy (0 = leaf)
            conn: Optional database connection to use

        Returns:
            List of created DocumentChunk instances
        """
        created_chunks = []

        # Use provided connection or create new transaction
        if conn:
            for idx, (content, metadata) in enumerate(chunks):
                chunk = await self._create_single_chunk(
                    conn,
                    document_id,
                    content,
                    metadata,
                    parent_chunk_id,
                    chunk_level,
                    idx,
                )
                if chunk:
                    created_chunks.append(chunk)
        else:
            async with self.db.db_manager.transaction() as conn:
                for idx, (content, metadata) in enumerate(chunks):
                    chunk = await self._create_single_chunk(
                        conn,
                        document_id,
                        content,
                        metadata,
                        parent_chunk_id,
                        chunk_level,
                        idx,
                    )
                    if chunk:
                        created_chunks.append(chunk)

        logger.info(f"Added {len(created_chunks)} chunks to document {document_id}")
        return created_chunks

    async def _create_single_chunk(
        self,
        conn: Any,
        document_id: UUID,
        content: str,
        metadata: Dict[str, Any],
        parent_chunk_id: Optional[UUID],
        chunk_level: int,
        chunk_index: int,
    ) -> Optional[DocumentChunk]:
        """Create a single chunk."""
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Count tokens (simplified - in production use tiktoken)
        token_count = len(content.split())

        chunk = DocumentChunk(
            document_id=document_id,
            parent_chunk_id=parent_chunk_id,
            chunk_index=chunk_index,
            chunk_level=chunk_level,
            content=content,
            content_hash=content_hash,
            token_count=token_count,
            metadata=metadata,
        )

        # Check if chunk already exists for this document
        query = self.db.db_manager._prepare_query(
            "SELECT chunk_id FROM {{tables.document_chunks}} WHERE document_id = $1 AND content_hash = $2"
        )
        existing = await conn.fetchrow(query, str(document_id), content_hash)

        if existing:
            logger.debug(f"Chunk already exists with hash {content_hash}")
            chunk.chunk_id = UUID(str(existing["chunk_id"]))
        else:
            # Insert new chunk
            query = self.db.db_manager._prepare_query(
                """
            INSERT INTO {{tables.document_chunks}} (
                chunk_id, document_id, parent_chunk_id, chunk_index,
                chunk_level, content, content_hash, token_count, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            RETURNING chunk_id, created_at
            """
            )

            result = await conn.fetchrow(
                query,
                str(chunk.chunk_id),
                str(chunk.document_id),
                str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
                chunk.chunk_index,
                chunk.chunk_level,
                chunk.content,
                chunk.content_hash,
                chunk.token_count,
                json.dumps(chunk.metadata),
            )

            chunk.created_at = result["created_at"]

            # Queue for embedding generation
            await self._queue_embedding_job(chunk.chunk_id, conn=conn)

        return chunk

    async def add_chunks_with_details(
        self, document_id: UUID, chunks: List[DocumentChunk], conn: Optional[Any] = None
    ) -> List[DocumentChunk]:
        """
        Add chunks while preserving all their details including token counts.

        Args:
            document_id: Document ID
            chunks: List of DocumentChunk objects with all details
            conn: Optional database connection to use

        Returns:
            List of created DocumentChunk instances
        """
        created_chunks = []

        # Use provided connection or create new transaction
        if conn:
            for chunk in chunks:
                created_chunk = await self._store_chunk_with_details(conn, chunk)
                if created_chunk:
                    created_chunks.append(created_chunk)
        else:
            async with self.db.db_manager.transaction() as conn:
                for chunk in chunks:
                    created_chunk = await self._store_chunk_with_details(conn, chunk)
                    if created_chunk:
                        created_chunks.append(created_chunk)

        return created_chunks

    async def _store_chunk_with_details(
        self, conn: Any, chunk: DocumentChunk
    ) -> Optional[DocumentChunk]:
        """Store a single chunk with all its details."""
        # Check if chunk already exists for this document
        query = self.db.db_manager._prepare_query(
            "SELECT chunk_id FROM {{tables.document_chunks}} WHERE document_id = $1 AND content_hash = $2"
        )
        existing = await conn.fetchrow(query, str(chunk.document_id), chunk.content_hash)

        if existing:
            logger.debug(f"Chunk already exists with hash {chunk.content_hash}")
            chunk.chunk_id = UUID(str(existing["chunk_id"]))
        else:
            # Insert new chunk with all details preserved
            query = self.db.db_manager._prepare_query(
                """
            INSERT INTO {{tables.document_chunks}} (
                chunk_id, document_id, parent_chunk_id, chunk_index,
                chunk_level, content, content_hash, token_count, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            RETURNING chunk_id, created_at
            """
            )

            result = await conn.fetchrow(
                query,
                str(chunk.chunk_id),
                str(chunk.document_id),
                str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
                chunk.chunk_index,
                chunk.chunk_level,
                chunk.content,
                chunk.content_hash,
                chunk.token_count,  # Preserve the actual token count
                json.dumps(chunk.metadata),
            )

            chunk.created_at = result["created_at"]
            logger.debug(
                f"Added chunk {chunk.chunk_id} with {chunk.token_count} tokens"
            )

        return chunk

    async def _queue_embedding_job(
        self,
        chunk_id: UUID,
        provider_id: Optional[str] = None,
        conn: Optional[Any] = None,
    ) -> None:
        """Queue a chunk for embedding generation."""
        # Get default provider if not specified
        if provider_id is None:
            provider_query = self.db.db_manager._prepare_query(
                """
            SELECT provider_id
            FROM {{tables.embedding_providers}}
            WHERE is_default = true
            LIMIT 1
            """
            )
            if conn:
                result = await conn.fetchrow(provider_query)
            else:
                result = await self.db.db_manager.fetch_one(provider_query)
            if result:
                provider_id = result["provider_id"]
            else:
                raise ValueError(
                    "No default embedding provider configured. "
                    "Please configure a default provider in the embedding_providers table."
                )

        query = self.db.db_manager._prepare_query(
            """
        INSERT INTO {{tables.embedding_queue}} (chunk_id, provider_id)
        VALUES ($1, $2)
        ON CONFLICT (chunk_id, provider_id) DO NOTHING
        """
        )

        if conn:
            await conn.execute(query, str(chunk_id), provider_id)
        else:
            await self.db.db_manager.execute(query, str(chunk_id), provider_id)

    async def update_chunk_embedding(
        self, chunk_id: UUID, embedding: List[float], provider_id: Optional[str] = None
    ) -> None:
        """
        Update the embedding for a chunk.

        Args:
            chunk_id: Chunk ID
            embedding: Embedding vector (1536 dimensions for text-embedding-3-small)
            provider_id: Optional provider ID (defaults to system default)
        """
        # Validate embedding dimension against provider configuration
        if provider_id is None:
            provider_row = await self.db.db_manager.fetch_one(
                """
                SELECT dimension, provider_id
                FROM {{tables.embedding_providers}}
                WHERE is_default = true
                LIMIT 1
                """
            )
        else:
            provider_row = await self.db.db_manager.fetch_one(
                """
                SELECT dimension, provider_id
                FROM {{tables.embedding_providers}}
                WHERE provider_id = $1
                """,
                provider_id,
            )

        expected_dim = int(provider_row["dimension"]) if provider_row else 1536
        resolved_provider_id = provider_row["provider_id"] if provider_row else provider_id
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Expected embedding of dimension {expected_dim}, got {len(embedding)}"
            )

        async with self.db.db_manager.transaction() as conn:
            # Insert or update chunk embedding in provider-specific table
            success = await self.db.insert_chunk_embedding(
                str(chunk_id), embedding, resolved_provider_id, conn=conn
            )

            if not success:
                raise RuntimeError(f"Failed to update embedding for chunk {chunk_id}")

            # Update embedding queue status
            # If provider_id is None, get the default provider
            if resolved_provider_id is None:
                provider_query = self.db.db_manager._prepare_query(
                    """
                SELECT provider_id
                FROM {{tables.embedding_providers}}
                WHERE is_default = true
                LIMIT 1
                """
                )
                result = await conn.fetchrow(provider_query)
                if result:
                    resolved_provider_id = result["provider_id"]
                else:
                    raise ValueError(
                        "No default embedding provider configured. "
                        "Please configure a default provider in the embedding_providers table."
                    )

            queue_query = self.db.db_manager._prepare_query(
                """
            UPDATE {{tables.embedding_queue}}
            SET status = $1, processed_at = NOW()
            WHERE chunk_id = $2 AND provider_id = $3
            """
            )

            await conn.execute(
                queue_query, EmbeddingStatus.COMPLETED.value, str(chunk_id), resolved_provider_id
            )

    async def search(
        self, query: SearchQuery, query_embedding: Optional[List[float]] = None
    ) -> List[SearchResult]:
        """
        Perform search based on query parameters.

        Args:
            query: SearchQuery instance with search parameters
            query_embedding: Optional embedding vector for vector/hybrid search

        Returns:
            List of SearchResult instances
        """
        if query.search_type == "vector" and query_embedding:
            results = await self.db.search_similar_chunks(
                query.owner_id,
                query_embedding,
                limit=query.limit,
                metadata_filter=query.metadata_filter,
                id_at_origin=query.id_at_origin,
                id_at_origins=query.id_at_origins,
                date_from=query.date_from,
                date_to=query.date_to,
            )
        elif query.search_type == "hybrid" and query_embedding:
            results = await self.db.hybrid_search(
                query.owner_id,
                query.query_text,
                query_embedding,
                limit=query.limit,
                alpha=query.alpha,
                metadata_filter=query.metadata_filter,
                id_at_origins=query.id_at_origins,
                date_from=query.date_from,
                date_to=query.date_to,
            )
        else:
            # Fall back to text search
            results = await self._text_search(query)

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            # Parse metadata if it's a string
            metadata = result["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            search_result = SearchResult(
                chunk_id=UUID(str(result["chunk_id"])),
                document_id=UUID(str(result["document_id"])),
                content=result["content"],
                metadata=metadata,
                score=result.get(
                    "similarity", result.get("rrf_score", result.get("rank", 0))
                ),
                similarity=result.get("similarity"),
                text_rank=result.get("rank"),
                rrf_score=result.get("rrf_score"),
            )

            # Get parent context if requested
            if query.include_parent_context:
                search_result.parent_chunks = await self._get_parent_context(
                    search_result.chunk_id, query.context_window
                )

            search_results.append(search_result)

        # Log search for analytics
        await self._log_search(query, search_results, query_embedding)

        return search_results

    async def _text_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform full-text search."""
        # Build query dynamically
        query_parts = [
            """
        SELECT
            c.chunk_id,
            c.document_id,
            c.content,
            c.metadata,
            ts_rank_cd(c.search_vector, websearch_to_tsquery('english', $1)) as rank
        FROM {{tables.document_chunks}} c
        JOIN {{tables.documents}} d ON c.document_id = d.document_id
        WHERE c.search_vector @@ websearch_to_tsquery('english', $1)
        AND d.owner_id = $2
        """
        ]

        params = [query.query_text, query.owner_id]
        param_count = 3

        if query.metadata_filter:
            query_parts.append(f"AND c.metadata @> ${param_count}::jsonb")
            params.append(json.dumps(query.metadata_filter))
            param_count += 1

        # Support single or multiple origins
        if query.id_at_origins:
            query_parts.append(f"AND d.id_at_origin = ANY(${param_count})")
            params.append(query.id_at_origins)
            param_count += 1
        elif query.id_at_origin:
            query_parts.append(f"AND d.id_at_origin = ${param_count}")
            params.append(query.id_at_origin)
            param_count += 1

        # Date range filtering
        if query.date_from:
            query_parts.append(f"AND d.document_date >= ${param_count}")
            params.append(query.date_from)
            param_count += 1

        if query.date_to:
            query_parts.append(f"AND d.document_date <= ${param_count}")
            params.append(query.date_to)
            param_count += 1

        query_parts.append(f"ORDER BY rank DESC LIMIT ${param_count}")
        params.append(query.limit)

        text_query = "\n".join(query_parts)
        results = await self.db.db_manager.fetch_all(text_query, *params)

        return results

    async def _get_parent_context(
        self, chunk_id: UUID, context_window: int
    ) -> List[DocumentChunk]:
        """Get parent context for a chunk."""
        # Use template-based schema qualification for the function
        query = self.db.db_manager._prepare_query(
            """
            SELECT * FROM {{schema}}.get_chunk_with_context($1, $2)
            """
        )

        results = await self.db.db_manager.fetch_all(
            query, str(chunk_id), context_window
        )

        chunks = []
        for row in results:
            if not row["is_target"]:  # Skip target chunk
                chunk = DocumentChunk(
                    chunk_id=UUID(str(row["chunk_id"])),
                    content=row["content"],
                    chunk_level=row["chunk_level"],
                    chunk_index=row["chunk_index"],
                )
                chunks.append(chunk)

        return chunks

    async def _log_search(
        self,
        query: SearchQuery,
        results: List[SearchResult],
        query_embedding: Optional[List[float]] = None,
    ) -> None:
        """Log search for analytics."""
        # Get default provider for search logging
        provider_id = None
        if query.search_type in ["vector", "hybrid"] and query_embedding:
            provider_query = """
            SELECT provider_id
            FROM {{tables.embedding_providers}}
            WHERE is_default = true
            LIMIT 1
            """
            result = await self.db.db_manager.fetch_one(provider_query)
            if result:
                provider_id = result["provider_id"]

        insert_query = """
        INSERT INTO {{tables.search_history}} (
            owner_id, id_at_origin, query_text, provider_id, search_type,
            metadata_filter, result_count, results
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        try:
            await self.db.db_manager.execute(
                insert_query,
                query.owner_id,
                query.id_at_origin or "unknown",
                query.query_text,
                provider_id,
                query.search_type.value,
                json.dumps(query.metadata_filter) if query.metadata_filter else None,
                len(results),
                json.dumps([r.to_dict() for r in results[:5]]),  # Log top 5
            )
        except Exception as e:
            logger.error(f"Failed to log search: {e}")

    async def get_pending_embeddings(self, limit: int = 100) -> List[EmbeddingJob]:
        """Get pending embedding jobs."""
        query = self.db.db_manager._prepare_query(
            """
        SELECT
            eq.chunk_id, eq.provider_id, eq.status, eq.retry_count,
            eq.created_at, c.content
        FROM {{tables.embedding_queue}} eq
        JOIN {{tables.document_chunks}} c ON eq.chunk_id = c.chunk_id
        WHERE eq.status = $1
        ORDER BY eq.created_at
        LIMIT $2
        """
        )

        results = await self.db.db_manager.fetch_all(
            query, EmbeddingStatus.PENDING.value, limit
        )

        jobs = []
        for row in results:
            # Use provider_id from the row instead of non-existent queue_id
            job = EmbeddingJob(
                chunk_id=UUID(str(row["chunk_id"])),
                provider_id=row["provider_id"],
                status=EmbeddingStatus(row["status"]),
                retry_count=row.get("retry_count", 0),
                error_message=row.get("error_message"),
                created_at=row["created_at"],
                processed_at=row.get("processed_at"),
            )
            jobs.append(job)

        return jobs

    async def delete_document(self, document_id: UUID) -> None:
        """Delete a document and all its chunks."""
        query = """
        DELETE FROM {{tables.documents}}
        WHERE document_id = $1
        """

        await self.db.db_manager.execute(query, str(document_id))
        logger.info(f"Deleted document {document_id}")

    async def add_chunk(
        self,
        document_id: UUID,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_chunk_id: Optional[UUID] = None,
        chunk_level: int = 0,
    ) -> DocumentChunk:
        """
        Add a single chunk to a document.

        Args:
            document_id: Document ID
            content: Chunk content
            metadata: Optional metadata
            parent_chunk_id: Optional parent chunk ID
            chunk_level: Level in hierarchy (0 = leaf)

        Returns:
            Created DocumentChunk instance
        """
        chunks = await self.add_chunks(
            document_id=document_id,
            chunks=[(content, metadata or {})],
            parent_chunk_id=parent_chunk_id,
            chunk_level=chunk_level,
        )
        return chunks[0] if chunks else None

    async def get_document_chunks(self, document_id: UUID) -> List[DocumentChunk]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of DocumentChunk instances
        """
        query = self.db.db_manager._prepare_query(
            """
        SELECT
            chunk_id, document_id, parent_chunk_id, chunk_index,
            chunk_level, content, content_hash, token_count,
            metadata, created_at
        FROM {{tables.document_chunks}}
        WHERE document_id = $1
        ORDER BY chunk_index
        """
        )

        rows = await self.db.db_manager.fetch_all(query, str(document_id))

        chunks = []
        for row in rows:
            chunk = DocumentChunk(
                chunk_id=UUID(row["chunk_id"]),
                document_id=UUID(row["document_id"]),
                parent_chunk_id=(
                    UUID(row["parent_chunk_id"]) if row["parent_chunk_id"] else None
                ),
                chunk_index=row["chunk_index"],
                chunk_level=row["chunk_level"],
                content=row["content"],
                content_hash=row["content_hash"],
                token_count=row["token_count"],
                metadata=row["metadata"],
                created_at=row["created_at"],
            )
            chunks.append(chunk)

        return chunks

    async def close(self) -> None:
        """Close database connections."""
        if self._initialized:
            # Let MemoryDatabase handle whether to actually close based on external_db flag
            await self.db.close()
            self._initialized = False
