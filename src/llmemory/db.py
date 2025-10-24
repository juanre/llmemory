# ABOUTME: Database integration layer providing high-level async operations for llmemory using pgdbm.
# ABOUTME: Manages document storage, chunk embeddings, search operations, and migration management with schema isolation.

"""Async database integration for llmemory using pgdbm-utils."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pgdbm import (AsyncDatabaseManager, AsyncMigrationManager,
                      DatabaseConfig, MonitoredAsyncDatabaseManager)

logger = logging.getLogger(__name__)


async def create_memory_db_manager(
    connection_string: Optional[str] = None,
    schema: str = "llmemory",
    enable_pgvector: bool = True,
    enable_monitoring: bool = False,
    min_connections: int = 10,
    max_connections: int = 20,
) -> AsyncDatabaseManager:
    """
    Create an async database manager for llmemory.

    Args:
        connection_string: PostgreSQL connection string
        schema: Database schema name
        enable_pgvector: Whether to enable pgvector extension
        enable_monitoring: Whether to enable database monitoring
        min_connections: Minimum pool size
        max_connections: Maximum pool size

    Returns:
        Configured AsyncDatabaseManager instance
    """
    config = DatabaseConfig(
        connection_string=connection_string,
        schema=schema,
        min_connections=min_connections,
        max_connections=max_connections,
        command_timeout=60,
        max_queries=50000,
        max_inactive_connection_lifetime=300,
        server_settings={"jit": "off"},  # Disable JIT for more predictable performance
    )

    # Use monitored version if requested
    if enable_monitoring:
        db = MonitoredAsyncDatabaseManager(config)
    else:
        db = AsyncDatabaseManager(config)

    await db.connect()

    # Enable pgvector extension if requested
    if enable_pgvector:
        await _ensure_pgvector_extension(db)

    return db


async def _ensure_pgvector_extension(db: AsyncDatabaseManager) -> None:
    """Ensure pgvector extension is enabled in the database."""
    try:
        # Check if pgvector is already enabled
        result = await db.fetch_value(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )

        if not result:
            # Enable pgvector extension (requires superuser privileges)
            await db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector extension enabled successfully")

        # Verify vector operations actually work
        await db.fetch_value("SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector")

    except Exception as e:
        raise ValueError(
            f"pgvector extension not available or not working: {e}. "
            "Install pgvector extension: CREATE EXTENSION vector; "
            "Ensure the extension is properly installed in PostgreSQL."
        ) from e


class MemoryDatabase:
    """High-level async interface for llmemory database operations."""

    def __init__(self, db: AsyncDatabaseManager, external_db: bool = False):
        self.db = db
        self._initialized = False
        self._external_db = external_db
        self.migration_manager: Optional[AsyncMigrationManager] = None

    @property
    def db_manager(self) -> AsyncDatabaseManager:
        """Get the underlying AsyncDatabaseManager for low-level operations."""
        return self.db

    @classmethod
    def from_manager(
        cls, db_manager: AsyncDatabaseManager, schema: str = "llmemory"
    ) -> "MemoryDatabase":
        """
        Create MemoryDatabase instance from existing AsyncDatabaseManager.

        This is used when integrating with a shared connection pool.

        Args:
            db_manager: Existing AsyncDatabaseManager instance
            schema: Database schema to use

        Returns:
            MemoryDatabase instance configured for external db management
        """
        # Just use the provided db_manager directly
        # The schema should already be set correctly by the caller
        # This follows the pattern from the integration guide where
        # the parent creates: AsyncDatabaseManager(pool=shared_pool, schema="llmemory")
        return cls(db_manager, external_db=True)

    async def initialize(self) -> None:
        """Initialize database and prepare statements."""
        if self._initialized:
            return

        # Set up migration manager
        migrations_path = Path(__file__).parent / "migrations"
        self.migration_manager = AsyncMigrationManager(
            self.db, migrations_path=str(migrations_path), module_name="llmemory"
        )

        # Register prepared statements for performance
        self._register_prepared_statements()
        self._initialized = True

    def _register_prepared_statements(self) -> None:
        """Register frequently used queries as prepared statements."""
        # Insert chunk query
        self.db.add_prepared_statement(
            "insert_chunk",
            """
            INSERT INTO {{tables.document_chunks}} (
                document_id, chunk_id, content, metadata, parent_chunk_id,
                chunk_index, chunk_level, content_hash, token_count
            ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
            RETURNING chunk_id
            """,
        )

        # Search queries
        self.db.add_prepared_statement(
            "get_provider_info",
            """
            SELECT provider_id, table_name
            FROM {{tables.embedding_providers}}
            WHERE is_default = true
            LIMIT 1
            """,
        )

    def _qualify_table(self, table_name: str) -> str:
        """Qualify runtime-dynamic table name with schema.

        For use with embedding provider tables that are created dynamically at runtime.
        Static tables should use {{tables.tablename}} template syntax instead.
        """
        if self.db.schema and self.db.schema != "public":
            return f'"{self.db.schema}"."{table_name}"'
        return f'"{table_name}"'

    async def apply_migrations(self) -> Dict[str, Any]:
        """Apply database migrations."""
        if not self._initialized:
            await self.initialize()
        return await self.migration_manager.apply_pending_migrations()

    async def insert_chunk(
        self,
        document_id: str,
        chunk_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_chunk_id: Optional[str] = None,
        chunk_index: int = 0,
        chunk_level: int = 0,
        provider_id: Optional[str] = None,
    ) -> str:
        """Insert a document chunk and optionally its embedding."""
        if metadata is None:
            metadata = {}

        # Generate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Simple token count (in production use tiktoken)
        token_count = len(content.split())

        # Use transaction for atomicity
        async with self.db.transaction() as conn:
            # First insert the chunk without embedding
            query = self.db.prepare_query(
                """
                INSERT INTO {{tables.document_chunks}} (
                    document_id, chunk_id, content, metadata, parent_chunk_id,
                    chunk_index, chunk_level, content_hash, token_count
                ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                RETURNING chunk_id
                """
            )
            result = await conn.fetch_one(
                query,
                document_id,
                chunk_id,
                content,
                json.dumps(metadata),
                parent_chunk_id,
                chunk_index,
                chunk_level,
                content_hash,
                token_count,
            )

            returned_chunk_id = result["chunk_id"] if result else None

            # If embedding provided, insert it into the appropriate provider table
            if returned_chunk_id and embedding:
                await self.insert_chunk_embedding(
                    returned_chunk_id, embedding, provider_id, conn=conn
                )

        return returned_chunk_id

    async def insert_chunk_embedding(
        self,
        chunk_id: str,
        embedding: List[float],
        provider_id: Optional[str] = None,
        conn: Optional[Any] = None,
    ) -> bool:
        """Insert embedding for a chunk into the appropriate provider table."""
        # Use provided connection or get new one
        if conn:
            return await self._insert_embedding_with_conn(
                conn, chunk_id, embedding, provider_id
            )
        else:
            async with self.db.transaction() as tx:
                return await self._insert_embedding_with_conn(
                    tx, chunk_id, embedding, provider_id
                )

    async def _insert_embedding_with_conn(
        self,
        conn: Any,
        chunk_id: str,
        embedding: List[float],
        provider_id: Optional[str] = None,
    ) -> bool:
        """Internal method to insert embedding with specific connection."""
        # Get provider info
        if provider_id is None:
            # Use default provider
            query = self.db.prepare_query(
                """
                SELECT provider_id, table_name, dimension
                FROM {{tables.embedding_providers}}
                WHERE is_default = true
                LIMIT 1
                """
            )
            provider_info = await conn.fetch_one(query)
        else:
            query = self.db.prepare_query(
                """
                SELECT provider_id, table_name, dimension FROM {{tables.embedding_providers}}
                WHERE provider_id = $1
                """
            )
            provider_info = await conn.fetch_one(query, provider_id)

        if not provider_info:
            raise ValueError(f"Provider {provider_id or 'default'} not found")

        table_name = provider_info["table_name"]
        # Validate embedding dimension
        provider_dimension = int(provider_info.get("dimension", 0))
        if provider_dimension and len(embedding) != provider_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for provider {provider_info.get('provider_id')}: "
                f"expected {provider_dimension}, got {len(embedding)}"
            )

        # Qualify dynamic table name (runtime from database, not compile-time static)
        qualified_table = self._qualify_table(table_name)

        insert_query = f"""
        INSERT INTO {qualified_table} (chunk_id, embedding)
        VALUES ($1, $2::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """

        try:
            # Convert embedding list to PostgreSQL array format
            embedding_str = f"[{','.join(map(str, embedding))}]"
            await conn.execute(insert_query, chunk_id, embedding_str)
            return True
        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            return False

    async def search_similar_chunks(
        self,
        owner_id: str,
        query_embedding: List[float],
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        id_at_origin: Optional[str] = None,
        id_at_origins: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        # Get default provider info
        provider_info = await self.db.fetch_one(
            """
            SELECT table_name
            FROM {{tables.embedding_providers}}
            WHERE is_default = true
            LIMIT 1
            """
        )

        if not provider_info:
            return []  # No default provider configured

        embedding_table = provider_info["table_name"]
        qualified_embedding = self._qualify_table(embedding_table)

        # Build query with mix of static templates and dynamic qualified table
        query_parts = [
            f"""
        SELECT
            c.chunk_id,
            c.document_id,
            c.content,
            c.metadata,
            1 - (e.embedding <=> $1::vector) as similarity
        FROM {{{{tables.document_chunks}}}} c
        JOIN {{{{tables.documents}}}} d ON c.document_id = d.document_id
        JOIN {qualified_embedding} e ON c.chunk_id = e.chunk_id
        WHERE d.owner_id = $2
        """
        ]

        # Convert embedding list to PostgreSQL vector string format
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        params = [embedding_str, owner_id]
        param_count = 3

        # Add optional filters
        if metadata_filter:
            query_parts.append(f"AND c.metadata @> ${param_count}::jsonb")
            params.append(json.dumps(metadata_filter))
            param_count += 1

        if id_at_origins:
            query_parts.append(f"AND d.id_at_origin = ANY(${param_count})")
            params.append(id_at_origins)
            param_count += 1
        elif id_at_origin:
            query_parts.append(f"AND d.id_at_origin = ${param_count}")
            params.append(id_at_origin)
            param_count += 1

        if date_from:
            query_parts.append(f"AND d.document_date >= ${param_count}")
            params.append(date_from)
            param_count += 1

        if date_to:
            query_parts.append(f"AND d.document_date <= ${param_count}")
            params.append(date_to)
            param_count += 1

        query_parts.append("ORDER BY e.embedding <=> $1::vector")
        query_parts.append(f"LIMIT ${param_count}")
        params.append(limit)

        query = "\n".join(query_parts)
        results = await self.db.fetch_all(query, *params)

        return results

    async def hybrid_search(
        self,
        owner_id: str,
        query_text: str,
        query_embedding: List[float],
        limit: int = 10,
        alpha: float = 0.5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        id_at_origins: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and full-text search.

        Uses Reciprocal Rank Fusion (RRF) to combine results.
        Alpha parameter controls the weight: 0 = text only, 1 = vector only
        """
        # Vector search results
        vector_results = await self.search_similar_chunks(
            owner_id,
            query_embedding,
            limit=limit * 2,  # Get more candidates for fusion
            metadata_filter=metadata_filter,
            id_at_origins=id_at_origins,
            date_from=date_from,
            date_to=date_to,
        )

        # Build full-text search query
        text_query_parts = [
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

        params = [query_text, owner_id]
        param_count = 3

        if metadata_filter:
            text_query_parts.append(f"AND c.metadata @> ${param_count}::jsonb")
            params.append(json.dumps(metadata_filter))
            param_count += 1

        if id_at_origins:
            text_query_parts.append(f"AND d.id_at_origin = ANY(${param_count})")
            params.append(id_at_origins)
            param_count += 1

        if date_from:
            text_query_parts.append(f"AND d.document_date >= ${param_count}")
            params.append(date_from)
            param_count += 1

        if date_to:
            text_query_parts.append(f"AND d.document_date <= ${param_count}")
            params.append(date_to)
            param_count += 1

        text_query_parts.append(f"ORDER BY rank DESC LIMIT ${param_count}")
        params.append(limit * 2)

        text_query = "\n".join(text_query_parts)
        text_results = await self.db.fetch_all(text_query, *params)

        # Combine using RRF
        return self._reciprocal_rank_fusion(vector_results, text_results, alpha, limit)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        alpha: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Combine results using Reciprocal Rank Fusion."""
        k = 60  # RRF constant

        # Store results and scores separately
        chunk_data = {}
        rrf_scores = {}

        # Process vector results
        for i, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            vector_score = alpha / (k + i + 1)

            # Store result data if not seen before
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result.copy()

            # Update RRF score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + vector_score

        # Process text results
        for i, result in enumerate(text_results):
            chunk_id = result["chunk_id"]
            text_score = (1 - alpha) / (k + i + 1)

            # Store result data if not seen before
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result.copy()

            # Update RRF score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + text_score

        # Add RRF scores to results and sort
        results_with_scores = []
        for chunk_id, score in rrf_scores.items():
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = score
            results_with_scores.append(result)

        # Sort by RRF score
        sorted_results = sorted(
            results_with_scores, key=lambda x: x["rrf_score"], reverse=True
        )

        return sorted_results[:limit]

    # Utility methods for testing and high-level operations

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.

        Args:
            document_id: UUID of the document

        Returns:
            Document data as dictionary or None if not found
        """
        return await self.db.fetch_one(
            "SELECT * FROM {{tables.documents}} WHERE document_id = $1", document_id
        )

    async def get_document_chunks(
        self, document_id: str, include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: UUID of the document
            include_embeddings: Whether to join with embedding tables

        Returns:
            List of chunk dictionaries ordered by chunk_index
        """
        if include_embeddings:
            # Get default provider
            provider_info = await self.db.fetch_one(
                """
                SELECT table_name
                FROM {{tables.embedding_providers}}
                WHERE is_default = true
                LIMIT 1
                """
            )

            if provider_info:
                table_name = provider_info["table_name"]
                qualified_table = self._qualify_table(table_name)

                return await self.db.fetch_all(
                    f"""
                    SELECT c.*, e.embedding IS NOT NULL as has_embedding
                    FROM {{{{tables.document_chunks}}}} c
                    LEFT JOIN {qualified_table} e ON c.chunk_id = e.chunk_id
                    WHERE c.document_id = $1
                    ORDER BY c.chunk_index
                    """,
                    document_id,
                )

        return await self.db.fetch_all(
            """
            SELECT * FROM {{tables.document_chunks}}
            WHERE document_id = $1
            ORDER BY chunk_index
            """,
            document_id,
        )

    async def count_documents(self, owner_id: str) -> int:
        """
        Count documents for an owner.

        Args:
            owner_id: Owner identifier

        Returns:
            Number of documents
        """
        count = await self.db.fetch_value(
            "SELECT COUNT(*) FROM {{tables.documents}} WHERE owner_id = $1", owner_id
        )
        return int(count) if count else 0

    async def count_chunks(self, document_id: Optional[str] = None) -> int:
        """
        Count chunks, optionally for a specific document.

        Args:
            document_id: Optional document ID to filter by

        Returns:
            Number of chunks
        """
        if document_id:
            count = await self.db.fetch_value(
                "SELECT COUNT(*) FROM {{tables.document_chunks}} WHERE document_id = $1",
                document_id,
            )
        else:
            count = await self.db.fetch_value(
                "SELECT COUNT(*) FROM {{tables.document_chunks}}"
            )
        return int(count) if count else 0

    async def get_embedding_providers(self) -> List[Dict[str, Any]]:
        """
        Get all registered embedding providers.

        Returns:
            List of provider dictionaries
        """
        return await self.db.fetch_all(
            "SELECT * FROM {{tables.embedding_providers}} ORDER BY provider_id"
        )

    async def get_default_embedding_provider(self) -> Optional[Dict[str, Any]]:
        """
        Get the default embedding provider.

        Returns:
            Provider dictionary or None if no default
        """
        return await self.db.fetch_one(
            "SELECT * FROM {{tables.embedding_providers}} WHERE is_default = true LIMIT 1"
        )

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks (CASCADE).

        Args:
            document_id: UUID of the document

        Returns:
            True if document was deleted, False if not found
        """
        result = await self.db.execute(
            "DELETE FROM {{tables.documents}} WHERE document_id = $1", document_id
        )
        # asyncpg returns a string like "DELETE 1"
        return "DELETE 1" in result if result else False

    async def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists.

        Args:
            document_id: UUID of the document

        Returns:
            True if document exists
        """
        exists = await self.db.fetch_value(
            "SELECT EXISTS(SELECT 1 FROM {{tables.documents}} WHERE document_id = $1)",
            document_id,
        )
        return bool(exists)

    async def chunk_exists(self, chunk_id: str) -> bool:
        """
        Check if a chunk exists.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            True if chunk exists
        """
        exists = await self.db.fetch_value(
            "SELECT EXISTS(SELECT 1 FROM {{tables.document_chunks}} WHERE chunk_id = $1)",
            chunk_id,
        )
        return bool(exists)

    async def get_chunks_without_embeddings(
        self, limit: int = 100, provider_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chunks that don't have embeddings yet.

        Args:
            limit: Maximum number of chunks to return
            provider_id: Specific provider to check, or use default

        Returns:
            List of chunk dictionaries
        """
        # Get provider info
        if provider_id:
            provider_info = await self.db.fetch_one(
                "SELECT table_name FROM {{tables.embedding_providers}} WHERE provider_id = $1",
                provider_id,
            )
        else:
            provider_info = await self.get_default_embedding_provider()

        if not provider_info:
            return []

        table_name = provider_info["table_name"]
        qualified_table = self._qualify_table(table_name)

        return await self.db.fetch_all(
            f"""
            SELECT c.*
            FROM {{{{tables.document_chunks}}}} c
            LEFT JOIN {qualified_table} e ON c.chunk_id = e.chunk_id
            WHERE e.chunk_id IS NULL
            LIMIT $1
            """,
            limit,
        )

    async def test_vector_operations(self) -> bool:
        """
        Test that pgvector operations are working.

        Returns:
            True if vector operations work
        """
        try:
            # Test basic vector operation
            test_vec = [1.0, 0.0, 0.0] + [0.0] * 1533  # 1536 dimensions
            vec_str = f"[{','.join(map(str, test_vec))}]"

            result = await self.db.fetch_one(
                "SELECT $1::vector <=> $1::vector as distance", vec_str
            )

            # Distance to self should be 0
            return result is not None and float(result["distance"]) < 0.001
        except Exception as e:
            logger.warning(f"Vector operations test failed: {e}")
            return False

    async def clear_all_data(self, owner_id: str) -> Dict[str, int]:
        """
        Clear all data for a specific owner (useful for tests).

        Args:
            owner_id: Owner whose data to clear

        Returns:
            Dictionary with counts of deleted items
        """
        # Get counts before deletion
        doc_count = await self.count_documents(owner_id)

        # Delete documents (chunks cascade)
        await self.db.execute(
            "DELETE FROM {{tables.documents}} WHERE owner_id = $1", owner_id
        )

        return {"documents_deleted": doc_count}

    async def vector_similarity_direct(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors directly.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (1 - cosine distance)
        """
        vec1_str = f"[{','.join(map(str, embedding1))}]"
        vec2_str = f"[{','.join(map(str, embedding2))}]"

        result = await self.db.fetch_value(
            "SELECT 1 - ($1::vector <=> $2::vector) as similarity", vec1_str, vec2_str
        )

        return float(result) if result is not None else 0.0

    async def close(self) -> None:
        """Close database connection."""
        if self._initialized:
            # Only disconnect if we own the connection
            if not self._external_db:
                await self.db.disconnect()
            self._initialized = False
