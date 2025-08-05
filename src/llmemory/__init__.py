"""Aword Memory - Document memory system with vector search.

A standalone library that provides document ingestion, chunking, embedding generation,
and semantic search capabilities. Supports multi-tenant deployments through
PostgreSQL schema isolation.
"""

from .config import AwordMemoryConfig, get_config, set_config
from .exceptions import (AwordMemoryError, ChunkingError, ConfigurationError,
                         ConnectionError, DatabaseError, DocumentNotFoundError,
                         EmbeddingError, PermissionError, RateLimitError,
                         ResourceNotFoundError, SearchError, ValidationError)
from .library import AwordMemory
from .models import (ChunkingConfig, DeleteResult, Document, DocumentAddResult,
                     DocumentChunk, DocumentListResult, DocumentType,
                     DocumentWithChunks, EmbeddingJob, EmbeddingStatus,
                     EnrichedSearchResult, OwnerStatistics, SearchQuery,
                     SearchResult, SearchResultWithDocuments, SearchType)

__version__ = "0.2.0"
__all__ = [
    # Main interface
    "AwordMemory",
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
    "AwordMemoryConfig",
    "get_config",
    "set_config",
    # Exceptions
    "AwordMemoryError",
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
]
