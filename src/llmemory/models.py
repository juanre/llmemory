# ABOUTME: Data models defining core types for documents, chunks, search queries, and results with validation and serialization.
# ABOUTME: Provides type-safe dataclasses for all llmemory operations including metadata, embeddings, and search configuration.

"""Data models for llmemory."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    TEXT = "text"
    HTML = "html"
    DOCX = "docx"
    EMAIL = "email"
    REPORT = "report"
    CHAT = "chat"
    PRESENTATION = "presentation"  # PowerPoint, Google Slides
    LEGAL_DOCUMENT = "legal_document"  # Contracts, agreements
    TECHNICAL_DOC = "technical_doc"  # API docs, specifications
    BUSINESS_REPORT = "business_report"  # Financial reports, analytics
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    """Chunking strategies."""

    HIERARCHICAL = "hierarchical"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


class SearchType(str, Enum):
    """Search types."""

    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


class EmbeddingStatus(str, Enum):
    """Embedding generation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Document:
    """Document model."""

    document_id: UUID = field(default_factory=uuid4)
    owner_id: str = ""  # Owner identifier for filtering (e.g., workspace_id)
    id_at_origin: str = (
        ""  # User ID, thread ID, or other origin identifier within owner
    )
    document_type: DocumentType = DocumentType.UNKNOWN
    document_name: str = ""
    document_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "document_id": str(self.document_id),
            "id_at_origin": self.id_at_origin,
            "document_type": self.document_type.value,
            "document_name": self.document_name,
            "document_date": self.document_date,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class DocumentChunk:
    """Document chunk model."""

    chunk_id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    parent_chunk_id: Optional[UUID] = None
    chunk_index: int = 0
    chunk_level: int = 0  # 0 = leaf, 1 = section, 2 = chapter, etc.
    content: str = ""
    content_hash: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Embeddings are now stored in provider-specific tables
    # Remove the embedding field from here

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "parent_chunk_id": (
                str(self.parent_chunk_id) if self.parent_chunk_id else None
            ),
            "chunk_index": self.chunk_index,
            "chunk_level": self.chunk_level,
            "content": self.content,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    chunk_size: int = 1000  # tokens
    chunk_overlap: int = 200  # tokens
    min_chunk_size: int = 100  # tokens
    max_chunk_size: int = 2000  # tokens

    # Hierarchical chunking parameters
    enable_hierarchical: bool = True
    section_markers: List[str] = field(default_factory=lambda: ["#", "##", "###"])

    # Document-type specific settings
    document_type_configs: Dict[DocumentType, Dict[str, Any]] = field(
        default_factory=lambda: {
            DocumentType.PDF: {
                "preserve_pages": True,
                "extract_tables": True,
                "extract_images": False,
            },
            DocumentType.CODE: {
                "preserve_functions": True,
                "preserve_classes": True,
                "language_specific": True,
            },
            DocumentType.MARKDOWN: {
                "preserve_headers": True,
                "preserve_lists": True,
                "preserve_code_blocks": True,
            },
        }
    )


@dataclass
class SearchQuery:
    """Search query model."""

    owner_id: str  # Required owner filter
    query_text: str
    search_type: SearchType = SearchType.HYBRID
    limit: int = 10
    metadata_filter: Optional[Dict[str, Any]] = None
    id_at_origin: Optional[str] = None
    id_at_origins: Optional[List[str]] = None  # Support multiple origins within owner
    include_parent_context: bool = True
    context_window: int = 2

    # Date range filtering
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

    # Hybrid search parameters
    alpha: float = 0.5  # 0 = text only, 1 = vector only
    rerank: bool = False
    rerank_model: Optional[str] = None

    # Provider selection (optional, uses default if not specified)
    embedding_provider: Optional[str] = None


@dataclass
class SearchResult:
    """Search result model."""

    chunk_id: UUID
    document_id: UUID
    content: str
    metadata: Dict[str, Any]
    score: float  # Similarity score or rank

    # Optional fields populated based on search type
    similarity: Optional[float] = None  # Vector similarity
    text_rank: Optional[float] = None  # Full-text search rank
    rrf_score: Optional[float] = None  # Reciprocal Rank Fusion score

    # Parent context if requested
    parent_chunks: List["DocumentChunk"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "similarity": self.similarity,
            "text_rank": self.text_rank,
            "rrf_score": self.rrf_score,
            "parent_chunks": [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "content": chunk.content,
                    "chunk_level": chunk.chunk_level,
                }
                for chunk in self.parent_chunks
            ],
        }


@dataclass
class EmbeddingJob:
    """Embedding generation job."""

    queue_id: UUID = field(default_factory=uuid4)
    chunk_id: UUID = field(default_factory=uuid4)
    provider_id: str = ""  # Which provider to use
    status: EmbeddingStatus = EmbeddingStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None


@dataclass
class SearchHistoryEntry:
    """Search history entry for analytics."""

    search_id: UUID = field(default_factory=uuid4)
    owner_id: str = ""  # Required for multi-tenancy
    id_at_origin: str = ""
    query_text: str = ""
    provider_id: Optional[str] = None  # Which provider was used
    search_type: SearchType = SearchType.HYBRID
    metadata_filter: Optional[Dict[str, Any]] = None
    result_count: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    feedback: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DocumentListResult:
    """Result of listing documents."""

    documents: List[Document]
    total: int
    limit: int
    offset: int


@dataclass
class DocumentWithChunks:
    """Document with optional chunks."""

    document: Document
    chunks: Optional[List[DocumentChunk]]
    chunk_count: int


@dataclass
class EnrichedSearchResult(SearchResult):
    """Search result with document metadata."""

    document_name: str = ""
    document_type: str = ""
    document_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResultWithDocuments:
    """Search results with document metadata included."""

    results: List[EnrichedSearchResult]
    total: int


@dataclass
class DocumentAddResult:
    """Result of adding a document."""

    document: Document
    chunks_created: int
    embeddings_created: int
    processing_time_ms: float


@dataclass
class OwnerStatistics:
    """Statistics for an owner's documents."""

    document_count: int
    chunk_count: int
    total_size_bytes: int  # Estimated
    document_type_breakdown: Optional[Dict[DocumentType, int]] = None
    created_date_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class DeleteResult:
    """Result of batch delete operation."""

    deleted_count: int
    deleted_document_ids: List[UUID]
