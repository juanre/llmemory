# ABOUTME: High-level async library interface providing the main public API for llmemory document operations.
# ABOUTME: Implements document storage, search, deletion, and embedding management with comprehensive validation and error handling.

"""Main library interface for llmemory.

This module provides the public API for integrating llmemory into any application.
It handles async operations with clean interfaces and supports
multi-tenant configurations through owner_id filtering.
"""

import asyncio
import json
import logging
import time
from dataclasses import replace as dc_replace
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from uuid import UUID

from pgdbm import AsyncDatabaseManager

from .batch_processor import BatchEmbeddingProcessor
from .config import EmbeddingProviderConfig, LLMemoryConfig, apply_hnsw_profile, get_config
from .embedding_providers import EmbeddingProvider, EmbeddingProviderFactory
from .embeddings import EmbeddingGenerator
from .exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    PermissionError as LLMemoryPermissionError,
    ValidationError,
)
from .manager import MemoryManager
from .models import (
    ChunkingConfig,
    DeleteResult,
    Document,
    DocumentAddResult,
    DocumentChunk,
    DocumentListResult,
    DocumentType,
    DocumentWithChunks,
    EmbeddingJob,
    EmbeddingStatus,
    EnrichedSearchResult,
    OwnerStatistics,
    SearchQuery,
    SearchResult,
    SearchResultWithDocuments,
    SearchType,
)
from .query_expansion import ExpansionCallback, QueryExpansionService
from .reranker import CrossEncoderReranker, OpenAIResponsesReranker, RerankerService
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
    ) -> None:
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
            raise ConfigurationError("Either connection string or db_manager is required")

        self.connection_string = connection_string
        self.openai_api_key = openai_api_key
        self.config = config or get_config()
        apply_hnsw_profile(self.config)
        self.validator = get_validator()
        self._external_db = db_manager is not None
        self._db_manager = db_manager

        # Update OpenAI API key in config if provided
        if self.openai_api_key and "openai" in self.config.embedding.providers:
            self.config.embedding.providers["openai"].api_key = self.openai_api_key

        self._manager: Optional[MemoryManager] = None
        self._embedding_providers: Dict[str, EmbeddingProvider] = {}
        self._batch_processor: Optional[BatchEmbeddingProcessor] = None
        self._background_stop_event: Optional[asyncio.Event] = None
        self._background_task: Optional[asyncio.Task] = None
        self._optimized_search: Optional[OptimizedAsyncSearch] = None
        self._reranker: Optional[RerankerService] = None
        self._query_expander: Optional[QueryExpansionService] = None
        self._summary_generator = self._create_summary_generator()
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
            if self._db_manager is None:
                raise ConfigurationError("db_manager is required when using external db mode")
            manager = MemoryManager.from_db_manager(
                self._db_manager,
                summary_generator=self._summary_generator,
                hnsw_ef_search=self.config.search.hnsw_ef_search,
                hnsw_m=self.config.database.hnsw_m,
                hnsw_ef_construction=self.config.database.hnsw_ef_construction,
            )
            await manager.initialize()
        else:
            # Create own db manager
            if self.connection_string is None:
                raise ConfigurationError("connection_string is required when not using external db")
            manager = await MemoryManager.create(
                self.connection_string,
                schema=self.config.database.schema_name,
                enable_monitoring=self.config.enable_metrics,
                min_connections=self.config.database.min_pool_size,
                max_connections=self.config.database.max_pool_size,
                summary_generator=self._summary_generator,
                hnsw_ef_search=self.config.search.hnsw_ef_search,
                hnsw_m=self.config.database.hnsw_m,
                hnsw_ef_construction=self.config.database.hnsw_ef_construction,
            )

        self._manager = manager

        # Initialize embedding providers registry
        await self._ensure_providers_registered()

        # Initialize optimized search
        self._optimized_search = OptimizedAsyncSearch(
            db=manager.db.db_manager,
            cache_ttl=self.config.search.cache_ttl,
            max_concurrent_queries=100,
            enable_query_optimization=True,
            hnsw_ef_search=self.config.search.hnsw_ef_search,
        )

        # Initialize query expansion and reranking
        expansion_callback = self._create_query_expansion_callback()
        self._query_expander = QueryExpansionService(
            self.config.search, llm_callback=expansion_callback
        )
        self._reranker = self._create_reranker_service()

        self._initialized = True
        logger.info("LLMemory initialized successfully")

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        await self.stop_background_processing()
        # Note: BatchEmbeddingProcessor has no close() method - no resources to clean up
        manager = self._manager
        if manager:
            await manager.close()

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
        Add or re-index a document with idempotent upsert semantics.

        Uses the archive-protocol identity contract:
        - owner_id = archive-protocol entity (e.g., jro, tsm, gsk)
        - id_at_origin = archive-protocol document_id

        Re-indexing the same (owner_id, id_at_origin) is idempotent:
        - Document record is preserved (same llmemory document_id)
        - Old chunks are replaced with new ones
        - No duplicates are created

        Args:
            owner_id: Archive-protocol entity identifier (e.g., jro, tsm, gsk)
            id_at_origin: Archive-protocol document_id (stable origin identifier)
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
        manager = await self._ensure_initialized()

        start_time = time.time()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)
        self.validator.validate_id_at_origin(id_at_origin)
        self.validator.validate_document_name(document_name)

        if isinstance(document_type, str):
            document_type = DocumentType(document_type)

        # Process the document
        doc, chunks = await manager.process_document(
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

        # Mark chunks as contextualized if feature is enabled
        if self.config.chunking.enable_contextual_retrieval and chunks:
            await self._mark_chunks_as_contextualized(chunks)

        embeddings_created = 0
        # Generate embeddings if requested
        if generate_embeddings and chunks:
            embeddings_created = await self._generate_embeddings_for_chunks(chunks, doc)

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
        query_expansion: Optional[bool] = None,
        max_query_variants: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
        rerank_return_k: Optional[int] = None,
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
            query_expansion: Override for query expansion (None = follow config)
            max_query_variants: Override for max query variants when expansion enabled
            rerank: Override for reranking (None = follow config)
            rerank_top_k: Candidate count for reranker consideration
            rerank_return_k: Preferred results prioritized by reranker

        Returns:
            List of SearchResult instances

        Raises:
            ValidationError: If input validation fails
            SearchError: If search operation fails
        """
        manager = await self._ensure_initialized()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)
        self.validator.validate_query_text(query_text)

        if isinstance(search_type, str):
            search_type = SearchType(search_type)

        # Determine expansion behaviour
        expansion_enabled = (
            query_expansion
            if query_expansion is not None
            else self.config.search.enable_query_expansion
        )
        max_variants = max_query_variants or self.config.search.max_query_variants
        if max_variants <= 0:
            max_variants = 1

        # Determine reranking behaviour
        should_rerank = rerank if rerank is not None else self.config.search.enable_rerank
        should_rerank = bool(should_rerank and self._reranker)

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
            enable_query_expansion=expansion_enabled,
            max_query_variants=max_variants,
            rerank=should_rerank,
            rerank_model=self.config.search.default_rerank_model,
            rerank_top_k=rerank_top_k or self.config.search.rerank_top_k,
            rerank_return_k=max(limit, rerank_return_k or self.config.search.rerank_return_k),
        )

        variants = await self._generate_query_variants(search_query)
        search_query.query_variants = variants

        if len(variants) == 1:
            if not should_rerank:
                # Generate query embedding if needed
                query_embedding = None
                if search_type in [SearchType.VECTOR, SearchType.HYBRID]:
                    query_embedding = await self._generate_query_embedding(query_text)

                return await manager.search(search_query, query_embedding)

            return await self._single_query_with_rerank(search_query)

        return await self._multi_query_search(search_query, variants, should_rerank)

    def _create_summary_generator(self) -> Optional[Callable[[DocumentChunk], Optional[str]]]:
        """Create a lightweight summary generator if enabled in config."""
        if not self.config.chunking.enable_chunk_summaries:
            return None

        max_tokens = max(10, self.config.chunking.summary_max_tokens)
        max_words = max(10, max_tokens // 2)

        def generator(chunk: DocumentChunk) -> Optional[str]:
            text = chunk.content.strip()
            if not text:
                return None

            words = text.split()
            summary_words = words[:max_words]
            summary = " ".join(summary_words)
            if len(words) > max_words:
                summary += "..."

            # Provide a hint from metadata if available
            title = chunk.metadata.get("title") if isinstance(chunk.metadata, dict) else None
            if title and title not in summary:
                summary = f"{title}: {summary}"

            return summary

        return generator

    def _create_query_expansion_callback(self) -> Optional[ExpansionCallback]:
        """Create LLM callback for query expansion if configured.

        Returns:
            Async callback function that generates query variants using LLM,
            or None if no expansion model configured.
        """
        model = self.config.search.query_expansion_model
        if not model:
            return None

        # Check if we have OpenAI API key
        if not self.openai_api_key:
            logger.warning(
                "query_expansion_model configured but no OpenAI API key available. "
                "Falling back to heuristic expansion."
            )
            return None

        async def openai_expansion_callback(query_text: str, max_variants: int) -> Sequence[str]:
            """Generate query variants using OpenAI."""
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.openai_api_key)

            prompt = f"""Generate {max_variants} alternative search queries that capture the same intent as the original query.

Original query: {query_text}

Requirements:
1. Semantically similar but use different words and phrasings
2. Include both more specific and more general variations
3. Capture different aspects or perspectives of the query
4. Keep queries concise (under 20 words each)
5. Return ONLY the alternative queries, one per line, no numbering or formatting

Alternative queries:"""

            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a query expansion expert. Generate diverse, semantically similar search queries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    timeout=5.0,
                )

                # Parse variants from response
                raw_content = response.choices[0].message.content
                if not raw_content:
                    return []
                content = raw_content.strip()
                variants = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip() and not line.strip().startswith(("#", "-", "*"))
                ]

                return variants[:max_variants]

            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}")
                return []

        return openai_expansion_callback

    def _create_reranker_service(self) -> RerankerService:
        """Instantiate reranker service with optional cross-encoder backend."""

        score_callback: Optional[
            Callable[[str, Sequence[SearchResult]], Awaitable[Sequence[float]]]
        ] = None

        provider = (self.config.search.rerank_provider or "lexical").lower()
        model_name = self.config.search.default_rerank_model or ""

        if provider == "openai":
            if not self.openai_api_key:
                logger.warning(
                    "rerank_provider='openai' configured but no OpenAI API key available. "
                    "Falling back to lexical reranker."
                )
            else:
                try:
                    reranker = OpenAIResponsesReranker(
                        model=model_name or "gpt-4.1-mini",
                        max_candidates=self.config.search.rerank_top_k,
                        api_key=self.openai_api_key,
                    )

                    async def callback(
                        query_text: str, results: Sequence[SearchResult]
                    ) -> Sequence[float]:
                        return await reranker.score(query_text, results)

                    score_callback = callback
                    logger.info("OpenAI reranker initialised with model %s", reranker.model)
                except ImportError as exc:
                    logger.warning("OpenAI reranker unavailable: %s. Falling back to lexical.", exc)
                except ValueError as exc:
                    logger.error("OpenAI reranker configuration error: %s", exc)

        elif provider in ["cross-encoder", "local"] or model_name.startswith("cross-encoder/"):
            try:
                cross_encoder = CrossEncoderReranker(
                    model_name=model_name,
                    device=self.config.search.rerank_device,
                    batch_size=self.config.search.rerank_batch_size,
                )

                async def callback(
                    query_text: str, results: Sequence[SearchResult]
                ) -> Sequence[float]:
                    return await cross_encoder.score(query_text, results)

                score_callback = callback
                logger.info("Cross-encoder reranker initialised: %s", model_name)
            except ImportError as exc:
                logger.warning(
                    "Failed to load cross-encoder '%s': %s. Falling back to lexical reranker.",
                    model_name,
                    exc,
                )
            except ValueError as exc:
                logger.error("Cross-encoder configuration error for '%s': %s", model_name, exc)

        return RerankerService(self.config.search, score_callback=score_callback)

    async def _generate_query_variants(self, query: SearchQuery) -> List[str]:
        """Generate query variants when expansion is enabled."""
        if not query.enable_query_expansion or not self._query_expander:
            return [query.query_text]

        variant_limit = max(1, query.max_query_variants)
        additional_needed = variant_limit - 1
        if additional_needed <= 0:
            return [query.query_text]

        expansions = await self._query_expander.expand(
            query.query_text,
            max_variants=additional_needed,
            include_keyword_variant=self.config.search.include_keyword_variant,
        )

        ordered = [query.query_text]
        seen = {query.query_text.lower()}
        for variant in expansions:
            normalized = variant.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(normalized)
            if len(ordered) >= variant_limit:
                break

        return ordered[:variant_limit]

    async def _multi_query_search(
        self, base_query: SearchQuery, variants: List[str], apply_rerank: bool
    ) -> List[SearchResult]:
        """Execute multi-query retrieval with RRF fusion."""
        manager = await self._ensure_initialized()

        total_start = time.perf_counter()
        variant_results: List[List[SearchResult]] = []
        variant_stats: List[Dict[str, Any]] = []
        embedding_required = base_query.search_type in [SearchType.VECTOR, SearchType.HYBRID]
        embedding_cache: Dict[str, List[float]] = {}

        for variant in variants:
            variant_query = dc_replace(
                base_query,
                query_text=variant,
                enable_query_expansion=False,
                query_variants=variants,
            )

            variant_start = time.perf_counter()
            query_embedding = None
            if embedding_required:
                if variant in embedding_cache:
                    query_embedding = embedding_cache[variant]
                else:
                    query_embedding = await self._generate_query_embedding(variant)
                    embedding_cache[variant] = query_embedding

            results = await manager.search(
                variant_query,
                query_embedding=query_embedding,
                disable_logging=True,
            )

            latency_ms = (time.perf_counter() - variant_start) * 1000
            variant_results.append(results)
            variant_stats.append(
                {
                    "query": variant,
                    "latency_ms": round(latency_ms, 3),
                    "result_count": len(results),
                }
            )

        fused_results = self._fuse_variant_results(variant_results, base_query.limit)

        rerank_info: Dict[str, Any] = {}
        if apply_rerank:
            fused_results, rerank_info = await self._apply_rerank(base_query, fused_results)

        total_latency_ms = (time.perf_counter() - total_start) * 1000
        diagnostics: Dict[str, Any] = {
            "latency_ms": round(total_latency_ms, 3),
            "backend": "multi_query",
            "search_type": (
                base_query.search_type.value
                if hasattr(base_query.search_type, "value")
                else str(base_query.search_type)
            ),
            "result_count": len(fused_results),
            "query_embedding_used": embedding_required,
            "rerank_requested": base_query.rerank,
            "rerank_applied": rerank_info.get("rerank_applied", False),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_variants": variants,
            "variant_stats": variant_stats,
        }
        diagnostics.update(rerank_info)

        base_query.query_variants = variants

        await manager.log_search_results(base_query, fused_results, diagnostics=diagnostics)

        return fused_results

    async def _single_query_with_rerank(self, query: SearchQuery) -> List[SearchResult]:
        """Execute single query search with optional reranking."""
        manager = await self._ensure_initialized()

        search_start = time.perf_counter()
        query_embedding = None
        if query.search_type in [SearchType.VECTOR, SearchType.HYBRID]:
            query_embedding = await self._generate_query_embedding(query.query_text)

        results = await manager.search(query, query_embedding, disable_logging=True)
        search_latency_ms = (time.perf_counter() - search_start) * 1000

        rerank_info: Dict[str, Any] = {}
        if query.rerank:
            results, rerank_info = await self._apply_rerank(query, results)

        diagnostics = {
            "latency_ms": round(search_latency_ms + rerank_info.get("rerank_latency_ms", 0.0), 3),
            "backend": (
                query.search_type.value
                if hasattr(query.search_type, "value")
                else str(query.search_type)
            ),
            "search_latency_ms": round(search_latency_ms, 3),
            "result_count": len(results),
            "query_embedding_used": bool(query_embedding),
            "rerank_requested": query.rerank,
            "rerank_applied": rerank_info.get("rerank_applied", False),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_variants": [query.query_text],
        }
        diagnostics.update(rerank_info)

        await manager.log_search_results(query, results, query_embedding, diagnostics=diagnostics)

        return results

    async def _apply_rerank(
        self, query: SearchQuery, results: List[SearchResult]
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Apply reranking if configured."""
        if not query.rerank or not self._reranker:
            return results, {"rerank_applied": False}

        rerank_start = time.perf_counter()
        reranked = await self._reranker.rerank(
            query.query_text,
            results,
            top_k=query.rerank_top_k,
            return_k=query.rerank_return_k,
        )
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000

        info: Dict[str, Any] = {
            "rerank_applied": True,
            "rerank_latency_ms": round(rerank_latency_ms, 3),
            "rerank_model": query.rerank_model,
            "rerank_top_k": query.rerank_top_k,
            "rerank_return_k": query.rerank_return_k,
        }
        return reranked, info

    def _fuse_variant_results(
        self, variant_results: List[List[SearchResult]], limit: int
    ) -> List[SearchResult]:
        """Fuse variant results using Reciprocal Rank Fusion."""
        rrf_constant = max(1, self.config.search.rrf_k)
        max_candidates = max(limit * 2, 20)

        score_map: Dict[UUID, float] = {}
        chunk_lookup: Dict[UUID, SearchResult] = {}

        for results in variant_results:
            for rank, result in enumerate(results[:max_candidates]):
                weight = 1.0 / (rrf_constant + rank + 1)
                score_map[result.chunk_id] = score_map.get(result.chunk_id, 0.0) + weight
                if result.chunk_id not in chunk_lookup:
                    chunk_lookup[result.chunk_id] = result

        sorted_chunks = sorted(score_map.items(), key=lambda item: item[1], reverse=True)

        fused_results: List[SearchResult] = []
        for chunk_id, score in sorted_chunks[:limit]:
            base_result = chunk_lookup[chunk_id]
            fused_results.append(dc_replace(base_result, score=score, rrf_score=score))

        return fused_results

    async def delete_document(
        self,
        owner_id: str,
        document_id: Union[UUID, str],
    ) -> None:
        """Delete a document and all its chunks.

        This is owner-scoped: callers must supply `owner_id` to prevent accidental cross-tenant
        deletes when only a UUID is known.
        """
        manager = await self._ensure_initialized()

        self.validator.validate_owner_id(owner_id)

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        doc_row = await manager.db.get_document(str(document_id))
        if not doc_row:
            raise DocumentNotFoundError(document_id)

        if doc_row["owner_id"] != owner_id:
            raise LLMemoryPermissionError(
                "Document does not belong to owner",
                resource=str(document_id),
                action="delete_document",
            )

        await manager.delete_document(document_id)

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
        manager = await self._ensure_initialized()

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

        params: List[Any] = [owner_id]
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
        rows = await manager.db.db.fetch_all(query, *params)

        # Get total count
        count_query_parts = [
            """
        SELECT COUNT(*) as count
        FROM {{tables.documents}}
        WHERE owner_id = $1
        """
        ]

        count_params: List[Any] = [owner_id]

        if document_type:
            count_query_parts.append("AND document_type = $2")
            count_params.append(document_type.value)

        if metadata_filter:
            count_query_parts.append(f"AND metadata @> ${len(count_params) + 1}::jsonb")
            count_params.append(json.dumps(metadata_filter))

        count_query = "\n".join(count_query_parts)
        count_result = await manager.db.db.fetch_one(count_query, *count_params)
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

        return DocumentListResult(documents=documents, total=total, limit=limit, offset=offset)

    async def get_document(
        self,
        owner_id: str,
        document_id: Union[str, UUID],
        include_chunks: bool = False,
        include_embeddings: bool = False,
    ) -> DocumentWithChunks:
        """
        Retrieve a specific document with optional chunks.

        Args:
            owner_id: Owner identifier (required for access control)
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
        manager = await self._ensure_initialized()

        self.validator.validate_owner_id(owner_id)

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        # Get document from database
        doc_row = await manager.db.get_document(str(document_id))

        if not doc_row:
            raise DocumentNotFoundError(document_id)

        if doc_row["owner_id"] != owner_id:
            raise LLMemoryPermissionError(
                "Document does not belong to owner",
                resource=str(document_id),
                action="get_document",
            )

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
        chunk_count = await manager.db.count_chunks(str(document_id))

        # Get chunks if requested
        chunks = None
        if include_chunks:
            chunk_rows = await manager.db.get_document_chunks(
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
                        UUID(str(row["parent_chunk_id"])) if row["parent_chunk_id"] else None
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

        return DocumentWithChunks(document=document, chunks=chunks, chunk_count=chunk_count)

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
        manager = await self._ensure_initialized()

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
            doc_ids = list({str(result.document_id) for result in search_results})

            # Fetch document metadata in batch - use the manager's prepared query
            doc_query = manager.db.db.prepare_query(
                """
            SELECT document_id, document_name, document_type, metadata
            FROM {{tables.documents}}
            WHERE document_id = ANY($1::uuid[])
              AND owner_id = $2
            """
            )
            doc_rows = await manager.db.db.fetch_all(doc_query, doc_ids, owner_id)

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

        return SearchResultWithDocuments(results=enriched_results, total=len(enriched_results))

    async def search_with_routing(
        self,
        owner_id: str,
        query_text: str,
        enable_routing: bool = True,
        routing_threshold: float = 0.7,
        **search_kwargs: Any,
    ) -> Dict[str, Any]:
        """Search with automatic query routing.

        Args:
            owner_id: Owner identifier
            query_text: Search query
            enable_routing: Enable query routing (default: True)
            routing_threshold: Confidence threshold for routing
            **search_kwargs: Additional arguments for search()

        Returns:
            Dict with:
            - route: RouteType (retrieval, web_search, unanswerable, clarification)
            - confidence: float (0-1)
            - results: List[SearchResult] (if route=retrieval)
            - message: str (if route != retrieval)
        """
        if not enable_routing:
            # Direct search without routing
            results = await self.search(owner_id, query_text, **search_kwargs)
            return {"route": "retrieval", "confidence": 1.0, "results": results}

        # Check for OpenAI API key before attempting routing
        if not self.openai_api_key:
            logger.warning(
                "Query routing requires OpenAI API key. "
                "Falling back to direct retrieval. "
                "Set OPENAI_API_KEY environment variable to enable routing."
            )
            results = await self.search(owner_id, query_text, **search_kwargs)
            return {
                "route": "retrieval",
                "confidence": 0.5,
                "results": results,
                "reason": "Query routing unavailable (no API key)",
            }

        # Get sample documents for routing context
        sample_docs = await self.list_documents(owner_id, limit=5)
        context = [doc.document_name for doc in sample_docs.documents]

        # Create router
        from .query_router import QueryRouter, RouteType

        router = QueryRouter(openai_api_key=self.openai_api_key, model="gpt-4o-mini")

        # Route query
        decision = await router.route(query_text, context, routing_threshold)

        if decision.route_type == RouteType.RETRIEVAL:
            results = await self.search(owner_id, query_text, **search_kwargs)
            return {
                "route": "retrieval",
                "confidence": decision.confidence,
                "results": results,
                "reason": decision.reason,
            }
        elif decision.route_type == RouteType.WEB_SEARCH:
            return {
                "route": "web_search",
                "confidence": decision.confidence,
                "message": "This query requires current or external information not in your documents.",
                "reason": decision.reason,
            }
        elif decision.route_type == RouteType.UNANSWERABLE:
            return {
                "route": "unanswerable",
                "confidence": decision.confidence,
                "message": "I cannot answer this type of query.",
                "reason": decision.reason,
            }
        else:  # CLARIFICATION
            return {
                "route": "clarification",
                "confidence": decision.confidence,
                "message": "Could you please provide more details about your question?",
                "reason": decision.reason,
            }

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
        manager = await self._ensure_initialized()

        # Validate inputs
        self.validator.validate_owner_id(owner_id)

        # Get document count
        doc_count = await manager.db.count_documents(owner_id)

        # Get chunk count and total size estimate
        chunk_query = """
        SELECT
            COUNT(*) as chunk_count,
            SUM(token_count * 4) as estimated_bytes
        FROM {{tables.document_chunks}} c
        JOIN {{tables.documents}} d ON c.document_id = d.document_id
        WHERE d.owner_id = $1
        """
        chunk_result = await manager.db.db.fetch_one(chunk_query, owner_id)

        chunk_count = chunk_result["chunk_count"] if chunk_result else 0
        total_size_bytes = int(chunk_result["estimated_bytes"] or 0) if chunk_result else 0

        # Get document type breakdown if requested
        type_breakdown = None
        if include_breakdown:
            breakdown_query = """
            SELECT document_type, COUNT(*) as count
            FROM {{tables.documents}}
            WHERE owner_id = $1
            GROUP BY document_type
            """
            breakdown_rows = await manager.db.db.fetch_all(breakdown_query, owner_id)

            type_breakdown = {
                DocumentType(row["document_type"]): row["count"] for row in breakdown_rows
            }

        # Get date range
        date_range_query = """
        SELECT
            MIN(created_at) as min_date,
            MAX(created_at) as max_date
        FROM {{tables.documents}}
        WHERE owner_id = $1
        """
        date_result = await manager.db.db.fetch_one(date_range_query, owner_id)

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
        owner_id: str,
        document_id: Union[str, UUID],
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[DocumentChunk]:
        """
        Get chunks for a specific document with pagination.

        Args:
            owner_id: Owner identifier (required for access control)
            document_id: Document identifier
            limit: Maximum number of chunks to return
            offset: Number of chunks to skip

        Returns:
            List of DocumentChunk objects

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        manager = await self._ensure_initialized()

        self.validator.validate_owner_id(owner_id)

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        # Validate pagination parameters
        if limit is not None and limit < 0:
            raise ValidationError("limit", "Limit must be non-negative")
        if offset < 0:
            raise ValidationError("offset", "Offset must be non-negative")

        # Check that document exists and belongs to the caller
        doc_row = await manager.db.get_document(str(document_id))
        if not doc_row:
            raise DocumentNotFoundError(document_id)

        if doc_row["owner_id"] != owner_id:
            raise LLMemoryPermissionError(
                "Document does not belong to owner",
                resource=str(document_id),
                action="get_document_chunks",
            )

        # Build query - use prepared query for schema support
        query = manager.db.db.prepare_query(
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

        params: List[Any] = [str(document_id)]

        if limit is not None:
            query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
            params.extend([limit, offset])

        rows = await manager.db.db.fetch_all(query, *params)

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
                    UUID(str(row["parent_chunk_id"])) if row["parent_chunk_id"] else None
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

    async def get_chunk_count(self, owner_id: str, document_id: Union[str, UUID]) -> int:
        """
        Get the number of chunks for a document.

        Args:
            owner_id: Owner identifier (required for access control)
            document_id: Document identifier

        Returns:
            Number of chunks

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        manager = await self._ensure_initialized()

        self.validator.validate_owner_id(owner_id)

        if isinstance(document_id, str):
            document_id = UUID(document_id)

        doc_row = await manager.db.get_document(str(document_id))
        if not doc_row:
            raise DocumentNotFoundError(document_id)

        if doc_row["owner_id"] != owner_id:
            raise LLMemoryPermissionError(
                "Document does not belong to owner",
                resource=str(document_id),
                action="get_chunk_count",
            )

        return await manager.db.count_chunks(str(document_id))

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
        manager = await self._ensure_initialized()

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
                doc_row = await manager.db.get_document(str(doc_id))
                if doc_row and doc_row["owner_id"] == owner_id:
                    await manager.delete_document(doc_id)
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
            rows = await manager.db.db.fetch_all(query, owner_id, json.dumps(metadata_filter))

            # Delete each matching document
            for row in rows:
                doc_id = UUID(str(row["document_id"]))
                await manager.delete_document(doc_id)
                deleted_ids.append(doc_id)

        return DeleteResult(deleted_count=len(deleted_ids), deleted_document_ids=deleted_ids)

    # Embedding Management

    async def process_pending_embeddings(self, batch_size: int = 100, max_batches: int = 10) -> int:
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
        manager = await self._ensure_initialized()

        total_processed = 0
        max_to_claim = max(1, int(batch_size) * max(1, int(max_batches)))

        jobs = await manager.claim_pending_embeddings(limit=max_to_claim)
        if not jobs:
            return 0

        jobs_by_provider: Dict[str, List[EmbeddingJob]] = {}
        for job in jobs:
            jobs_by_provider.setdefault(job.provider_id, []).append(job)

        for provider_id, provider_jobs in jobs_by_provider.items():
            generator = await self._get_embedding_generator_for_db_provider(provider_id)
            per_batch = max(1, int(generator.provider.config.batch_size or batch_size))

            for i in range(0, len(provider_jobs), per_batch):
                batch_jobs = provider_jobs[i : i + per_batch]
                chunk_texts = await self._fetch_embedding_texts(batch_jobs)

                # Preserve order of jobs for embedding generation
                texts: List[str] = []
                usable_jobs: List[EmbeddingJob] = []
                for job in batch_jobs:
                    text = chunk_texts.get(job.chunk_id)
                    if text is None:
                        await manager.update_embedding_job_status(
                            job.queue_id,
                            EmbeddingStatus.FAILED,
                            error_message="Chunk content not found for embedding job",
                            increment_retry=True,
                        )
                        continue
                    texts.append(text)
                    usable_jobs.append(job)

                if not usable_jobs:
                    continue

                try:
                    embeddings = await generator.generate_embeddings(texts)
                except Exception as exc:  # noqa: BLE001
                    for job in usable_jobs:
                        await manager.update_embedding_job_status(
                            job.queue_id,
                            EmbeddingStatus.FAILED,
                            error_message=str(exc),
                            increment_retry=True,
                        )
                    continue

                for job, embedding in zip(usable_jobs, embeddings, strict=True):
                    try:
                        await manager.update_chunk_embedding(
                            job.chunk_id, embedding, provider_id=provider_id
                        )
                        total_processed += 1
                    except Exception as exc:  # noqa: BLE001
                        await manager.update_embedding_job_status(
                            job.queue_id,
                            EmbeddingStatus.FAILED,
                            error_message=str(exc),
                            increment_retry=True,
                        )

        return total_processed

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

        await self.stop_background_processing()

        stop_event = asyncio.Event()
        self._background_stop_event = stop_event

        async def loop() -> None:
            while not stop_event.is_set():
                try:
                    processed = await self.process_pending_embeddings(
                        batch_size=batch_size, max_batches=1
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Background embedding processing failed: %s", exc)
                    processed = 0

                sleep_for = 1.0 if processed else float(interval_seconds)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                except asyncio.TimeoutError:
                    continue

        self._background_task = asyncio.create_task(loop())

    async def stop_background_processing(self) -> None:
        """Stop background embedding processing, if running."""
        if not self._background_task:
            return

        if self._background_stop_event:
            self._background_stop_event.set()

        task = self._background_task
        self._background_task = None
        self._background_stop_event = None

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Private Methods

    async def _ensure_initialized(self) -> MemoryManager:
        """Ensure the library is initialized and return the manager."""
        if not self._initialized:
            await self.initialize()

        if self._manager is None:
            raise RuntimeError("LLMemory is not initialized")

        return self._manager

    async def _ensure_providers_registered(self) -> None:
        """Ensure embedding providers are registered in database."""
        # This would typically register providers in the embedding_providers table
        # For now, we'll skip this as it's handled by migrations
        pass

    async def _get_embedding_generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator."""
        provider = await self._get_embedding_provider()
        return EmbeddingGenerator(provider)

    async def _get_embedding_generator_for_db_provider(
        self, provider_id: str
    ) -> EmbeddingGenerator:
        """Create or reuse an embedding generator for a provider registered in the database."""
        if provider_id in self._embedding_providers:
            return EmbeddingGenerator(self._embedding_providers[provider_id])

        manager = await self._ensure_initialized()

        row = await manager.db.db_manager.fetch_one(
            """
            SELECT provider_type, model_name, dimension
            FROM {{tables.embedding_providers}}
            WHERE provider_id = $1
            """,
            provider_id,
        )
        if not row:
            raise ConfigurationError(f"Embedding provider not found in database: {provider_id}")

        provider_type = str(row["provider_type"])
        model_name = str(row["model_name"])
        dimension = int(row["dimension"])

        provider_config = EmbeddingProviderConfig(
            provider_type=provider_type,
            model_name=model_name,
            dimension=dimension,
        )

        # Fill in provider-specific settings from the library config where possible.
        if provider_type == "openai":
            base = self.config.embedding.providers.get("openai")
            provider_config.api_key = self.openai_api_key or (base.api_key if base else None)
            if base:
                provider_config.batch_size = base.batch_size
                provider_config.max_retries = base.max_retries
                provider_config.retry_delay = base.retry_delay
                provider_config.timeout = base.timeout
                provider_config.max_requests_per_minute = base.max_requests_per_minute
                provider_config.max_tokens_per_minute = base.max_tokens_per_minute
        elif provider_type == "local":
            base = next(
                (
                    cfg
                    for cfg in self.config.embedding.providers.values()
                    if cfg.provider_type == "local" and cfg.model_name == model_name
                ),
                None,
            )
            if base is None:
                base = next(
                    (
                        cfg
                        for cfg in self.config.embedding.providers.values()
                        if cfg.provider_type == "local"
                    ),
                    None,
                )
            if base:
                provider_config.device = base.device
                provider_config.cache_dir = base.cache_dir
                provider_config.batch_size = base.batch_size
                provider_config.max_retries = base.max_retries
                provider_config.retry_delay = base.retry_delay
                provider_config.timeout = base.timeout

        provider = EmbeddingProviderFactory.create_provider(provider_id, provider_config)
        self._embedding_providers[provider_id] = provider
        return EmbeddingGenerator(provider)

    async def _fetch_embedding_texts(self, jobs: Sequence[EmbeddingJob]) -> Dict[UUID, str]:
        """Fetch (and optionally contextualize) chunk texts for embedding generation."""
        if not jobs:
            return {}
        manager = await self._ensure_initialized()

        chunk_ids = [str(job.chunk_id) for job in jobs]
        rows = await manager.db.db_manager.fetch_all(
            """
            SELECT
                c.chunk_id,
                c.content,
                c.metadata,
                d.document_name,
                d.document_type
            FROM {{tables.document_chunks}} c
            JOIN {{tables.documents}} d ON c.document_id = d.document_id
            WHERE c.chunk_id = ANY($1::uuid[])
            """,
            chunk_ids,
        )

        result: Dict[UUID, str] = {}
        for row in rows:
            chunk_id = UUID(str(row["chunk_id"]))
            content = row["content"] or ""
            metadata = row.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:  # noqa: BLE001
                    metadata = {}

            if (
                self.config.chunking.enable_contextual_retrieval
                and isinstance(metadata, dict)
                and metadata.get("contextualized")
            ):
                document_name = row.get("document_name") or ""
                document_type = row.get("document_type") or ""
                content = self.config.chunking.context_template.format(
                    document_name=document_name,
                    document_type=str(document_type),
                    content=content,
                )

            result[chunk_id] = content

        return result

    async def _get_embedding_provider(self, provider_id: Optional[str] = None) -> EmbeddingProvider:
        """Get embedding provider instance."""
        if provider_id is None:
            provider_id = self.config.embedding.default_provider

        if provider_id not in self._embedding_providers:
            provider_config = self.config.embedding.providers.get(provider_id)
            if not provider_config:
                raise ConfigurationError(f"Provider {provider_id} not configured")

            self._embedding_providers[provider_id] = EmbeddingProviderFactory.create_provider(
                provider_id, provider_config
            )

        return self._embedding_providers[provider_id]

    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query."""
        generator = await self._get_embedding_generator()
        return await generator.generate_embedding(query_text)

    async def _mark_chunks_as_contextualized(self, chunks: List[DocumentChunk]) -> None:
        """Mark chunks as contextualized in metadata during chunking phase.

        Sets the metadata flag in both memory and database so embeddings can be
        generated with context later, even if embeddings aren't created immediately.
        """
        manager = await self._ensure_initialized()

        for chunk in chunks:
            chunk.metadata["contextualized"] = True

            # Update metadata in database
            await manager.db.db_manager.execute(
                """
                UPDATE {{tables.document_chunks}}
                SET metadata = metadata || $1::jsonb
                WHERE chunk_id = $2
                """,
                json.dumps({"contextualized": True}),
                str(chunk.chunk_id),
            )

    def _contextualize_chunk(self, chunk: DocumentChunk, document: Document) -> Tuple[str, bool]:
        """Prepend document context to chunk for embedding.

        Returns tuple of (text_for_embedding, was_contextualized).
        Preserves original chunk.content for display.
        """
        if not self.config.chunking.enable_contextual_retrieval:
            return chunk.content, False

        # Format context template
        contextualized_text = self.config.chunking.context_template.format(
            document_name=document.document_name,
            document_type=document.document_type.value,
            content=chunk.content,
        )

        return contextualized_text, True

    async def _generate_embeddings_for_chunks(
        self, chunks: List[DocumentChunk], document: Optional[Document] = None
    ) -> int:
        """Generate embeddings for a list of chunks. Returns count of successful embeddings."""
        generator = await self._get_embedding_generator()
        manager = await self._ensure_initialized()
        successful_count = 0

        for chunk in chunks:
            try:
                # Check if chunk should be contextualized based on metadata flag
                text_for_embedding = chunk.content

                if document and chunk.metadata.get("contextualized"):
                    text_for_embedding, _ = self._contextualize_chunk(chunk, document)

                embedding = await generator.generate_embedding(text_for_embedding)
                await manager.update_chunk_embedding(chunk.chunk_id, embedding)
                successful_count += 1
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")

        return successful_count

    # Context Manager Support

    async def __aenter__(self) -> "LLMemory":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
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
