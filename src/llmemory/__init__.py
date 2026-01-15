# ABOUTME: Main package entry point exporting public API for llmemory document memory system with vector search.
# ABOUTME: Provides MemoryLibrary class and all public models, exceptions, and configuration objects for easy import.

"""llmemory - Document memory system with vector search.

A standalone library that provides document ingestion, chunking, embedding generation,
and semantic search capabilities. Supports multi-tenant deployments through
PostgreSQL schema isolation.
"""

from .config import LLMemoryConfig, get_config, set_config
from .exceptions import (
    ChunkingError,
    ConfigurationError,
    ConnectionError,
    DatabaseError,
    DocumentNotFoundError,
    EmbeddingError,
    LLMemoryError,
    PermissionError,
    RateLimitError,
    ResourceNotFoundError,
    SearchError,
    ValidationError,
)
from .library import LLMemory
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
from .query_router import QueryRouter, RouteDecision, RouteType
from .reranker import CrossEncoderReranker, OpenAIResponsesReranker

__version__ = "0.5.0"
__all__ = [
    # Main interface
    "LLMemory",
    # Models
    "Document",
    "DocumentChunk",
    "SearchQuery",
    "SearchResult",
    "DocumentType",
    "SearchType",
    "ChunkingConfig",
    "EmbeddingStatus",
    "EmbeddingJob",
    "DocumentListResult",
    "DocumentWithChunks",
    "EnrichedSearchResult",
    "SearchResultWithDocuments",
    "DocumentAddResult",
    "OwnerStatistics",
    "DeleteResult",
    # Configuration
    "LLMemoryConfig",
    "get_config",
    "set_config",
    # Exceptions
    "LLMemoryError",
    "ValidationError",
    "ConfigurationError",
    "DatabaseError",
    "EmbeddingError",
    "SearchError",
    "ChunkingError",
    "ResourceNotFoundError",
    "RateLimitError",
    "ConnectionError",
    "DocumentNotFoundError",
    "PermissionError",
    # Rerankers
    "CrossEncoderReranker",
    "OpenAIResponsesReranker",
    # Query Routing
    "QueryRouter",
    "RouteDecision",
    "RouteType",
]
