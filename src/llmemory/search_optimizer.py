"""Search optimization for high-performance queries.

Based on task-engine integration requirements:
- Search latency < 100ms (p95)
- API throughput: 1000 req/s
- Concurrent query handling
- Result caching with Redis
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pgdbm import AsyncDatabaseManager

from .models import DocumentChunk, SearchQuery, SearchResult, SearchType

logger = logging.getLogger(__name__)

# Prometheus metrics - optional import
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary

    # Define metrics
    search_requests_total = Counter(
        "aword_memory_search_requests_total",
        "Total number of search requests",
        ["search_type", "owner_id"],
    )

    search_duration_seconds = Histogram(
        "aword_memory_search_duration_seconds",
        "Search request duration in seconds",
        ["search_type"],
        buckets=(
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ),
    )

    search_results_count = Histogram(
        "aword_memory_search_results_count",
        "Number of results returned per search",
        ["search_type"],
        buckets=(0, 1, 5, 10, 20, 50, 100),
    )

    cache_hit_rate = Gauge("aword_memory_cache_hit_rate", "Cache hit rate for searches")

    active_searches = Gauge(
        "aword_memory_active_searches", "Number of currently active searches"
    )

    vector_similarity_scores = Summary(
        "aword_memory_vector_similarity_scores",
        "Distribution of vector similarity scores",
    )

    text_rank_scores = Summary(
        "aword_memory_text_rank_scores", "Distribution of text rank scores"
    )

    query_embedding_time = Histogram(
        "aword_memory_query_embedding_seconds",
        "Time to generate query embeddings",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )

    database_query_duration = Histogram(
        "aword_memory_database_query_seconds",
        "Database query execution time",
        ["query_type"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )

    METRICS_ENABLED = True
    logger.info("Prometheus metrics enabled for aword-memory")

except ImportError:
    # Fallback when prometheus_client is not installed
    METRICS_ENABLED = False
    logger.info(
        "Prometheus metrics disabled - install with: pip install aword-memory[monitoring]"
    )


@dataclass
class SearchCacheKey:
    """Cache key for search results."""

    owner_id: str
    query_text: str
    search_type: SearchType
    id_at_origins: Optional[List[str]] = None
    metadata_filter: Optional[Dict[str, Any]] = None

    def to_hash(self) -> str:
        """Generate a deterministic hash for the cache key."""
        key_data = {
            "owner_id": self.owner_id,
            "query_text": self.query_text,
            "search_type": self.search_type.value,
            "id_at_origins": sorted(self.id_at_origins or []),
            "metadata_filter": self.metadata_filter,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


@dataclass
class SearchMetrics:
    """Track search performance metrics."""

    query_count: int = 0
    total_latency_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.query_count == 0:
            return 0
        return self.total_latency_ms / self.query_count

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0
        return self.cache_hits / total


class OptimizedAsyncSearch:
    """Optimized search with caching and performance enhancements."""

    def __init__(
        self,
        db: AsyncDatabaseManager,
        cache_ttl: int = 300,  # 5 minutes
        max_concurrent_queries: int = 100,
        enable_query_optimization: bool = True,
    ):
        self.db = db
        self.cache_ttl = cache_ttl
        self.max_concurrent_queries = max_concurrent_queries
        self.enable_query_optimization = enable_query_optimization

        # In-memory cache (replace with Redis in production)
        self._cache: Dict[str, Tuple[List[Dict], datetime]] = {}
        self._cache_lock = asyncio.Lock()

        # Query semaphore for concurrency control
        self._query_semaphore = asyncio.Semaphore(max_concurrent_queries)

        # Metrics tracking
        self.metrics = SearchMetrics()

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform optimized search with caching and performance tracking.

        Args:
            query: SearchQuery instance

        Returns:
            List of SearchResult instances
        """
        start_time = time.time()

        # Increment active searches gauge
        if METRICS_ENABLED:
            active_searches.inc()
            search_requests_total.labels(
                search_type=query.search_type.value, owner_id=query.owner_id
            ).inc()

        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            cached_results = await self._get_cached_results(cache_key)

            if cached_results is not None:
                self.metrics.cache_hits += 1
                results = cached_results
            else:
                self.metrics.cache_misses += 1

                # Perform search with concurrency control
                async with self._query_semaphore:
                    if query.search_type == SearchType.VECTOR:
                        results = await self._optimized_vector_search(query)
                    elif query.search_type == SearchType.HYBRID:
                        results = await self._optimized_hybrid_search(query)
                    else:
                        results = await self._optimized_text_search(query)

                # Cache results
                await self._cache_results(cache_key, results)

            # Track metrics
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.query_count += 1
            self.metrics.total_latency_ms += latency_ms

            if METRICS_ENABLED:
                # Record search duration
                search_duration_seconds.labels(
                    search_type=query.search_type.value
                ).observe(time.time() - start_time)

                # Update cache hit rate gauge
                cache_hit_rate.set(self.metrics.cache_hit_rate)

            if latency_ms > 100:
                logger.warning(f"Search latency exceeded 100ms: {latency_ms:.2f}ms")

            # Convert to SearchResult objects
            search_results = await self._convert_to_search_results(results, query)

            if METRICS_ENABLED:
                # Record result count
                search_results_count.labels(
                    search_type=query.search_type.value
                ).observe(len(search_results))

            return search_results

        finally:
            # Decrement active searches gauge
            if METRICS_ENABLED:
                active_searches.dec()

    async def _optimized_vector_search(
        self, query: SearchQuery
    ) -> List[Dict[str, Any]]:
        """
        Optimized vector search with performance enhancements.

        Uses:
        - HNSW index for fast similarity search
        - Selective column loading
        - Query parallelization
        """
        # Build optimized query using HNSW index
        base_query = """
        WITH vector_search AS (
            SELECT
                c.chunk_id,
                c.document_id,
                c.content,
                c.metadata,
                c.chunk_level,
                1 - (c.embedding <=> %s::vector) as similarity
            FROM {{tables.document_chunks}} c
            JOIN {{tables.documents}} d ON c.document_id = d.document_id
            WHERE c.embedding IS NOT NULL
            AND d.owner_id = %s
            {filters}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        )
        SELECT * FROM vector_search WHERE similarity > 0.3
        """

        # Build filters
        filters = []
        params = [f"[{','.join(map(str, query.query_embedding))}]", query.owner_id]

        if query.metadata_filter:
            filters.append("AND c.metadata @> %s::jsonb")
            params.append(json.dumps(query.metadata_filter))

        if query.id_at_origins:
            filters.append("AND d.id_at_origin = ANY(%s)")
            params.append(query.id_at_origins)

        if query.date_from:
            filters.append("AND d.document_date >= %s")
            params.append(query.date_from)

        if query.date_to:
            filters.append("AND d.document_date <= %s")
            params.append(query.date_to)

        # Add chunk level optimization for hierarchical search
        if self.enable_query_optimization:
            # Prioritize higher-level chunks first
            filters.append("AND c.chunk_level >= 1")

        # Add embedding again for ORDER BY clause
        params.append(f"[{','.join(map(str, query.query_embedding))}]")
        params.append(query.limit * 2)  # Get extra for filtering

        final_query = base_query.replace("{filters}", " ".join(filters))

        # Track database query time
        db_start = time.time()
        results = await self.db.execute_and_fetch_all(final_query, tuple(params))

        if METRICS_ENABLED:
            database_query_duration.labels(query_type="vector_search").observe(
                time.time() - db_start
            )

            # Track similarity scores
            for row in results:
                vector_similarity_scores.observe(row["similarity"])

        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "metadata": row["metadata"],
                "chunk_level": row["chunk_level"],
                "similarity": row["similarity"],
            }
            for row in results
        ]

    async def _optimized_text_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """
        Optimized text search using GIN indexes.

        Uses:
        - GIN index for full-text search
        - Query rewriting for better results
        - Rank optimization
        """
        # Determine text search configuration from metadata
        # First, try to get language from query metadata
        text_config = "english"  # default
        if query.metadata_filter and "language" in query.metadata_filter:
            lang = query.metadata_filter["language"]
            # Map language to PostgreSQL config
            lang_configs = {
                "en": "english",
                "es": "spanish",
                "fr": "french",
                "de": "german",
                "it": "italian",
                "pt": "portuguese",
                "ru": "russian",
                "nl": "dutch",
                "ar": "arabic",
            }
            text_config = lang_configs.get(lang, "simple")

        # Use optimized query with GIN index hints and language-specific search
        text_query = f"""
        WITH text_search AS (
            SELECT
                c.chunk_id,
                c.document_id,
                c.content,
                c.metadata,
                c.chunk_level,
                ts_rank_cd(c.search_vector, query, 32) as rank
            FROM {{{{tables.document_chunks}}}} c
            JOIN {{{{tables.documents}}}} d ON c.document_id = d.document_id,
            websearch_to_tsquery('{text_config}', %s) query
            WHERE c.search_vector @@ query
            AND d.owner_id = %s
            {{filters}}
        )
        SELECT * FROM text_search
        WHERE rank > 0.01
        ORDER BY rank DESC
        LIMIT %s
        """

        # Build filters
        filters = []
        params = [query.query_text, query.owner_id]

        if query.metadata_filter:
            filters.append("AND c.metadata @> %s::jsonb")
            params.append(json.dumps(query.metadata_filter))

        if query.id_at_origins:
            filters.append("AND d.id_at_origin = ANY(%s)")
            params.append(query.id_at_origins)

        if query.date_from:
            filters.append("AND d.document_date >= %s")
            params.append(query.date_from)

        if query.date_to:
            filters.append("AND d.document_date <= %s")
            params.append(query.date_to)

        params.append(query.limit)

        # Replace filters placeholder
        final_query = text_query.replace("{filters}", " ".join(filters))

        # Debug: print query
        print(f"Text search query: {final_query}")

        # Track database query time
        db_start = time.time()
        results = await self.db.execute_and_fetch_all(final_query, tuple(params))

        if METRICS_ENABLED:
            database_query_duration.labels(query_type="text_search").observe(
                time.time() - db_start
            )

            # Track text rank scores
            for row in results:
                text_rank_scores.observe(row["rank"])

        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "metadata": row["metadata"],
                "chunk_level": row["chunk_level"],
                "rank": row["rank"],
            }
            for row in results
        ]

    async def _optimized_hybrid_search(
        self, query: SearchQuery
    ) -> List[Dict[str, Any]]:
        """
        Optimized hybrid search with parallel execution.

        Performs vector and text searches concurrently and merges results.
        """
        # Execute searches in parallel
        vector_task = asyncio.create_task(self._optimized_vector_search(query))
        text_task = asyncio.create_task(self._optimized_text_search(query))

        vector_results, text_results = await asyncio.gather(vector_task, text_task)

        # Apply optimized RRF
        return self._fast_reciprocal_rank_fusion(
            vector_results, text_results, query.alpha, query.limit
        )

    def _fast_reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        alpha: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fast RRF implementation with early termination."""
        k = 60  # RRF constant

        # Use dictionaries for O(1) lookups
        chunk_data = {}
        rrf_scores = {}

        # Process vector results
        for i, result in enumerate(vector_results[: limit * 2]):
            chunk_id = result["chunk_id"]
            vector_score = alpha / (k + i + 1)

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + vector_score

        # Process text results
        for i, result in enumerate(text_results[: limit * 2]):
            chunk_id = result["chunk_id"]
            text_score = (1 - alpha) / (k + i + 1)

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + text_score

        # Get top results efficiently
        top_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

        results = []
        for chunk_id, score in top_chunks:
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = score
            results.append(result)

        return results

    async def _convert_to_search_results(
        self, raw_results: List[Dict[str, Any]], query: SearchQuery
    ) -> List[SearchResult]:
        """Convert raw results to SearchResult objects with parent context."""
        search_results = []

        # Get parent contexts in batch if needed
        parent_contexts = {}
        if query.include_parent_context:
            chunk_ids = [UUID(r["chunk_id"]) for r in raw_results]
            parent_contexts = await self._batch_get_parent_contexts(
                chunk_ids, query.context_window
            )

        for result in raw_results:
            chunk_id = UUID(result["chunk_id"])

            search_result = SearchResult(
                chunk_id=chunk_id,
                document_id=UUID(result["document_id"]),
                content=result["content"],
                metadata=result["metadata"],
                score=result.get(
                    "similarity", result.get("rrf_score", result.get("rank", 0))
                ),
                similarity=result.get("similarity"),
                text_rank=result.get("rank"),
                rrf_score=result.get("rrf_score"),
            )

            # Add parent context if available
            if chunk_id in parent_contexts:
                search_result.parent_chunks = parent_contexts[chunk_id]

            search_results.append(search_result)

        return search_results

    async def _batch_get_parent_contexts(
        self, chunk_ids: List[UUID], context_window: int
    ) -> Dict[UUID, List[DocumentChunk]]:
        """Get parent contexts for multiple chunks in batch."""
        if not chunk_ids:
            return {}

        # Use a more efficient batch query
        query = """
        WITH target_chunks AS (
            SELECT unnest(%s::uuid[]) as chunk_id
        ),
        context_chunks AS (
            SELECT
                tc.chunk_id as target_id,
                c.chunk_id,
                c.content,
                c.chunk_level,
                c.chunk_index,
                c.parent_chunk_id,
                c.document_id
            FROM target_chunks tc
            JOIN {{tables.document_chunks}} target ON target.chunk_id = tc.chunk_id
            JOIN {{tables.document_chunks}} c ON c.document_id = target.document_id
            WHERE abs(c.chunk_index - target.chunk_index) <= %s
            AND c.chunk_id != tc.chunk_id
        )
        SELECT * FROM context_chunks
        ORDER BY target_id, chunk_index
        """

        results = await self.db.execute_and_fetch_all(
            query, ([str(cid) for cid in chunk_ids], context_window)
        )

        # Group by target chunk
        contexts = {}
        for row in results:
            target_id = UUID(row["target_id"])
            if target_id not in contexts:
                contexts[target_id] = []

            chunk = DocumentChunk(
                chunk_id=UUID(row["chunk_id"]),
                document_id=UUID(row["document_id"]),
                content=row["content"],
                chunk_level=row["chunk_level"],
                chunk_index=row["chunk_index"],
                parent_chunk_id=(
                    UUID(row["parent_chunk_id"]) if row["parent_chunk_id"] else None
                ),
            )
            contexts[target_id].append(chunk)

        return contexts

    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for a search query."""
        cache_key = SearchCacheKey(
            owner_id=query.owner_id,
            query_text=query.query_text,
            search_type=query.search_type,
            id_at_origins=query.id_at_origins,
            metadata_filter=query.metadata_filter,
        )
        return cache_key.to_hash()

    async def _get_cached_results(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached results if available and not expired."""
        async with self._cache_lock:
            if cache_key in self._cache:
                results, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return results
                else:
                    # Expired, remove from cache
                    del self._cache[cache_key]
        return None

    async def _cache_results(self, cache_key: str, results: List[Dict]) -> None:
        """Cache search results."""
        async with self._cache_lock:
            self._cache[cache_key] = (results, datetime.now())

            # Simple cache eviction if too large
            if len(self._cache) > 1000:
                # Remove oldest entries
                sorted_keys = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_keys[:100]:
                    del self._cache[key]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "query_count": self.metrics.query_count,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
        }

    async def warm_cache(self, common_queries: List[SearchQuery]) -> None:
        """Pre-warm cache with common queries."""
        logger.info(f"Warming cache with {len(common_queries)} queries")

        tasks = []
        for query in common_queries:
            task = asyncio.create_task(self.search(query))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Cache warm complete. Hit rate: {self.metrics.cache_hit_rate:.2%}")
