# ABOUTME: Configuration management providing centralized settings for chunking, embeddings, search, and validation parameters.
# ABOUTME: Supports environment variable overrides and provides type-safe configuration with sensible defaults for all components.

"""Configuration management for llmemory library."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EmbeddingProviderConfig:
    """Configuration for a single embedding provider."""

    provider_type: str  # "openai" or "local"
    model_name: str
    dimension: int

    # Provider-specific settings
    api_key: Optional[str] = None  # For OpenAI
    device: str = "cpu"  # For local models
    cache_dir: Optional[str] = None  # For local model downloads

    # Common settings
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    # Rate limiting (mainly for API providers)
    max_tokens_per_minute: int = 1_000_000
    max_requests_per_minute: int = 3_000


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    # Default provider to use
    default_provider: str = "openai"

    # Available providers
    providers: Dict[str, EmbeddingProviderConfig] = field(
        default_factory=lambda: {
            "openai": EmbeddingProviderConfig(
                provider_type="openai",
                model_name="text-embedding-3-small",
                dimension=1536,
            ),
            "local-minilm": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                device="cpu",
            ),
        }
    )

    # Whether to automatically create embedding tables for new providers
    auto_create_tables: bool = True


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    # Default chunk sizes (in tokens)
    default_parent_size: int = 1000
    default_child_size: int = 200
    default_overlap: int = 50

    # Summaries
    enable_chunk_summaries: bool = False
    summary_max_tokens: int = 120
    summary_prompt_template: str = (
        "Summarize the following text in one or two sentences focusing on key facts: {text}"
    )

    # Min/max constraints
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    max_chunk_depth: int = 3  # Maximum hierarchy depth

    # Document-type specific configurations
    chunk_configs: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "email": {"parent": 300, "child": 150, "overlap": 25},
            "business_report": {"parent": 600, "child": 300, "overlap": 50},
            "technical_doc": {"parent": 800, "child": 400, "overlap": 60},
            "presentation": {"parent": 400, "child": 200, "overlap": 30},
            "legal_document": {"parent": 500, "child": 250, "overlap": 40},
            "pdf": {"parent": 800, "child": 400, "overlap": 60},
            "markdown": {"parent": 600, "child": 300, "overlap": 50},
            "code": {"parent": 500, "child": 250, "overlap": 40},
        }
    )


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    # Search parameters
    default_limit: int = 10
    max_limit: int = 100
    default_search_type: str = "hybrid"

    # RRF parameters
    rrf_k: int = 50

    # Query expansion
    enable_query_expansion: bool = False
    max_query_variants: int = 3
    query_expansion_model: Optional[str] = None
    include_keyword_variant: bool = True

    # Reranking
    enable_rerank: bool = False
    default_rerank_model: Optional[str] = None
    rerank_top_k: int = 50
    rerank_return_k: int = 15

    # Vector search
    hnsw_ef_search: int = 100
    vector_search_limit: int = 100

    # Text search
    text_search_limit: int = 100
    text_search_config: str = "english"

    # Cache settings
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000

    # Performance
    search_timeout: float = 5.0
    min_score_threshold: float = 0.0


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""

    # Connection settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    connection_timeout: float = 10.0
    command_timeout: float = 30.0

    # Schema
    schema_name: str = "llmemory"
    documents_table: str = "documents"
    chunks_table: str = "document_chunks"
    embeddings_queue_table: str = "embedding_queue"
    search_history_table: str = "search_history"
    embedding_providers_table: str = "embedding_providers"
    chunk_embeddings_prefix: str = "chunk_embeddings_"

    # Index names
    hnsw_index_name: str = "document_chunks_embedding_hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


@dataclass
class ValidationConfig:
    """Configuration for input validation."""

    # Field constraints
    max_owner_id_length: int = 255
    max_id_at_origin_length: int = 255
    max_document_name_length: int = 500
    max_content_length: int = 10_000_000  # 10MB
    max_metadata_size: int = 65536  # 64KB

    # Required fields
    min_content_length: int = 10

    # Patterns
    valid_owner_id_pattern: str = r"^[a-zA-Z0-9_\-\.]+$"
    valid_id_at_origin_pattern: str = r"^[a-zA-Z0-9_\-\.@]+$"


@dataclass
class LLMemoryConfig:
    """Main configuration for llmemory library."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_background_processing: bool = True

    # Logging
    log_level: str = "INFO"
    log_slow_queries: bool = True
    slow_query_threshold: float = 1.0  # seconds

    @classmethod
    def from_env(cls) -> "LLMemoryConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        def env_var(*names: str) -> Optional[str]:
            """Return the first environment variable value found."""
            for name in names:
                value = os.getenv(name)
                if value is not None:
                    return value
            return None

        def env_bool(*names: str) -> Optional[bool]:
            raw = env_var(*names)
            if raw is None:
                return None
            return raw.lower() in {"1", "true", "yes", "on"}

        # Default provider
        if provider := env_var("LLMEMORY_EMBEDDING_PROVIDER", "AWORD_EMBEDDING_PROVIDER"):
            config.embedding.default_provider = provider

        # OpenAI configuration
        api_key = env_var("LLMEMORY_OPENAI_API_KEY", "AWORD_OPENAI_API_KEY", "OPENAI_API_KEY")
        if api_key:
            config.embedding.providers["openai"].api_key = api_key

        if model := env_var("LLMEMORY_OPENAI_MODEL", "AWORD_OPENAI_MODEL"):
            config.embedding.providers["openai"].model_name = model
            # Update dimension based on model
            if "text-embedding-3-small" in model:
                config.embedding.providers["openai"].dimension = 1536
            elif "text-embedding-3-large" in model:
                config.embedding.providers["openai"].dimension = 3072

        # Local model configuration
        if model := env_var("LLMEMORY_LOCAL_MODEL", "AWORD_LOCAL_MODEL"):
            config.embedding.providers["local-minilm"].model_name = model

        if device := env_var("LLMEMORY_LOCAL_DEVICE", "AWORD_LOCAL_DEVICE"):
            config.embedding.providers["local-minilm"].device = device

        if cache_dir := env_var("LLMEMORY_LOCAL_CACHE_DIR", "AWORD_LOCAL_CACHE_DIR"):
            config.embedding.providers["local-minilm"].cache_dir = cache_dir

        # Multiple providers support
        if providers_str := env_var("LLMEMORY_EMBEDDING_PROVIDERS", "AWORD_EMBEDDING_PROVIDERS"):
            # Parse comma-separated list of provider configurations
            # Format: provider1:type:model,provider2:type:model
            for provider_spec in providers_str.split(","):
                parts = provider_spec.strip().split(":")
                if len(parts) >= 3:
                    name, ptype, model = parts[0], parts[1], parts[2]
                    dimension = int(parts[3]) if len(parts) > 3 else None

                    provider_config = EmbeddingProviderConfig(
                        provider_type=ptype,
                        model_name=model,
                        dimension=dimension or (1536 if ptype == "openai" else 384),
                    )

                    # Copy API key for OpenAI providers
                    if ptype == "openai" and config.embedding.providers["openai"].api_key:
                        provider_config.api_key = config.embedding.providers["openai"].api_key

                    config.embedding.providers[name] = provider_config

        if cache_ttl := env_var("LLMEMORY_SEARCH_CACHE_TTL", "AWORD_SEARCH_CACHE_TTL"):
            config.search.cache_ttl = int(cache_ttl)

        if pool_size := env_var("LLMEMORY_DB_MAX_POOL_SIZE", "AWORD_DB_MAX_POOL_SIZE"):
            config.database.max_pool_size = int(pool_size)

        if log_level := env_var("LLMEMORY_LOG_LEVEL", "AWORD_LOG_LEVEL"):
            config.log_level = log_level

        # Feature flags
        if env_var("LLMEMORY_DISABLE_CACHING", "AWORD_DISABLE_CACHING"):
            config.enable_caching = False

        if env_var("LLMEMORY_DISABLE_METRICS", "AWORD_DISABLE_METRICS"):
            config.enable_metrics = False

        if (val := env_bool("LLMEMORY_ENABLE_QUERY_EXPANSION", "AWORD_ENABLE_QUERY_EXPANSION")) is not None:
            config.search.enable_query_expansion = val

        if (val := env_bool("LLMEMORY_ENABLE_RERANK", "AWORD_ENABLE_RERANK")) is not None:
            config.search.enable_rerank = val

        if (val := env_var("LLMEMORY_RERANK_TOP_K", "AWORD_RERANK_TOP_K")):
            config.search.rerank_top_k = int(val)

        if (val := env_var("LLMEMORY_RERANK_RETURN_K", "AWORD_RERANK_RETURN_K")):
            config.search.rerank_return_k = int(val)

        if (val := env_bool("LLMEMORY_ENABLE_CHUNK_SUMMARIES", "AWORD_ENABLE_CHUNK_SUMMARIES")) is not None:
            config.chunking.enable_chunk_summaries = val

        return config

    def validate(self) -> None:
        """Validate configuration values."""
        # Embedding validation
        if not self.embedding.providers:
            raise ValueError("At least one embedding provider must be configured")

        if self.embedding.default_provider not in self.embedding.providers:
            raise ValueError(
                f"Default provider '{self.embedding.default_provider}' not found in providers"
            )

        for name, provider in self.embedding.providers.items():
            if provider.dimension <= 0:
                raise ValueError(f"Embedding dimension for '{name}' must be positive")

            if provider.batch_size <= 0:
                raise ValueError(f"Batch size for '{name}' must be positive")

            if provider.provider_type == "openai" and not provider.api_key:
                # Check environment variable as fallback
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError(f"OpenAI provider '{name}' requires an API key")

        # Chunking validation
        if self.chunking.min_chunk_size >= self.chunking.max_chunk_size:
            raise ValueError("Min chunk size must be less than max chunk size")

        # Search validation
        if self.search.default_limit > self.search.max_limit:
            raise ValueError("Default limit cannot exceed max limit")

        # Database validation
        if self.database.min_pool_size > self.database.max_pool_size:
            raise ValueError("Min pool size cannot exceed max pool size")


# Global configuration instance
_config: Optional[LLMemoryConfig] = None


def get_config() -> LLMemoryConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = LLMemoryConfig.from_env()
        _config.validate()
    return _config


def set_config(config: LLMemoryConfig) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config
