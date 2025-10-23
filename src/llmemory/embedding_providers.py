# ABOUTME: Embedding provider implementations supporting OpenAI API and local Sentence Transformers models with rate limiting.
# ABOUTME: Handles API key management, request throttling, caching, and provider-specific configuration for embedding generation.

"""Embedding providers for aword-memory.

This module provides different embedding providers (OpenAI, local models, etc.)
with a common interface for generating embeddings.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from .config import EmbeddingProviderConfig, get_config
from .exceptions import ConfigurationError, EmbeddingError, RateLimitError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, provider_id: str, config: EmbeddingProviderConfig):
        """Initialize embedding provider.

        Args:
            provider_id: Unique identifier for this provider
            config: Provider configuration
        """
        self.provider_id = provider_id
        self.config = config
        self.table_name = self._get_table_name()

    def _get_table_name(self) -> str:
        """Get the table name for this provider's embeddings."""
        # Generate table name based on provider and dimension
        suffix = f"{self.config.provider_type}_{self.config.dimension}"
        if self.config.provider_type == "openai":
            return "chunk_embeddings_openai"
        else:
            # For local models, include model name in table
            model_short = self.config.model_name.split("/")[-1].lower()
            model_short = model_short.replace("-", "_").replace(".", "_")
            return f"chunk_embeddings_{model_short}_{self.config.dimension}"

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        return self.config.dimension

    def get_model_name(self) -> str:
        """Get the name of the model used by this provider."""
        return self.config.model_name

    def get_table_name(self) -> str:
        """Get the database table name for this provider's embeddings."""
        return self.table_name


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider using their API."""

    def __init__(self, provider_id: str, config: EmbeddingProviderConfig):
        """Initialize OpenAI embedding provider."""
        super().__init__(provider_id, config)

        if not config.api_key:
            # Try to get from environment
            import os

            config.api_key = os.getenv("OPENAI_API_KEY")
            if not config.api_key:
                raise ConfigurationError(
                    f"OpenAI provider '{provider_id}' requires an API key. "
                    "Set AWORD_OPENAI_API_KEY or OPENAI_API_KEY environment variable."
                )

        self._client = None
        self._request_times: List[datetime] = []
        self._semaphore = asyncio.Semaphore(10)  # Max concurrent requests

    def _ensure_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.config.api_key)

    async def _check_rate_limit(self, num_texts: int) -> None:
        """Check and enforce rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Remove old requests
        self._request_times = [t for t in self._request_times if t > minute_ago]

        # Check if we would exceed rate limit
        if len(self._request_times) + num_texts > self.config.max_requests_per_minute:
            # Calculate wait time
            oldest_request = self._request_times[0] if self._request_times else now
            wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record new requests
        self._request_times.extend([now] * num_texts)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []

        self._ensure_client()
        await self._check_rate_limit(len(texts))

        retry_count = 0
        while retry_count < self.config.max_retries:
            async with self._semaphore:
                try:
                    import openai

                    response = await self._client.embeddings.create(
                        model=self.config.model_name,
                        input=texts,
                        timeout=self.config.timeout,
                    )
                    embeddings = [e.embedding for e in response.data]

                    # Validate embedding dimensions
                    for i, emb in enumerate(embeddings):
                        if len(emb) != self.config.dimension:
                            raise EmbeddingError(
                                f"Unexpected embedding dimension: got {len(emb)}, "
                                f"expected {self.config.dimension}",
                                provider="openai",
                                text_index=i,
                            )

                    return embeddings

                except openai.RateLimitError as e:
                    retry_count += 1
                    if retry_count >= self.config.max_retries:
                        logger.error(
                            f"Rate limit exceeded after {self.config.max_retries} retries"
                        )
                        raise RateLimitError(
                            f"Rate limit exceeded: {str(e)}", retry_after=60.0
                        )

                    wait_time = min(
                        60 * retry_count, 300
                    )  # Exponential backoff, max 5 min
                    logger.warning(
                        f"Rate limit error (retry {retry_count}/{self.config.max_retries}), "
                        f"waiting {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)

                except openai.APIConnectionError as e:
                    logger.error(f"API connection error: {e}")
                    raise EmbeddingError(
                        f"Failed to connect to OpenAI API: {str(e)}",
                        provider="openai",
                        error_type="connection",
                    )

                except openai.AuthenticationError as e:
                    logger.error(f"Authentication error: {e}")
                    raise EmbeddingError(
                        f"OpenAI authentication failed: {str(e)}",
                        provider="openai",
                        error_type="authentication",
                    )

                except Exception as e:
                    logger.error(f"Unexpected error generating embeddings: {e}")
                    raise EmbeddingError(
                        f"Failed to generate embeddings: {str(e)}",
                        provider="openai",
                        error_type="unknown",
                    )


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embeddings provider using sentence-transformers."""

    def __init__(self, provider_id: str, config: EmbeddingProviderConfig):
        """Initialize local embedding provider."""
        super().__init__(provider_id, config)

        self._model = None
        self._dependencies_loaded = False
        self._sentence_transformers = None
        self._torch = None

    def _ensure_dependencies(self):
        """Lazily load dependencies."""
        if not self._dependencies_loaded:
            try:
                # Set environment variable before importing
                import os

                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # Import dependencies
                import sentence_transformers
                import torch

                self._sentence_transformers = sentence_transformers
                self._torch = torch
                self._dependencies_loaded = True

                logger.info("Local embedding dependencies loaded successfully")

            except ImportError as e:
                raise ImportError(
                    "Local embeddings require sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from e

    def _ensure_model_loaded(self):
        """Lazily load the model."""
        if self._model is None:
            self._ensure_dependencies()

            logger.info(f"Loading local embedding model: {self.config.model_name}")

            cache_folder = self.config.cache_dir
            if not cache_folder:
                import os

                cache_folder = os.getenv("TRANSFORMERS_CACHE")

            self._model = self._sentence_transformers.SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                cache_folder=cache_folder,
            )

            # Verify dimension
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.config.dimension:
                logger.warning(
                    f"Model {self.config.model_name} has dimension {actual_dim}, "
                    f"but config specifies {self.config.dimension}. Using actual dimension."
                )
                self.config.dimension = actual_dim

            logger.info(
                f"Local model loaded successfully, dimension: {self.config.dimension}, "
                f"device: {self.config.device}"
            )

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        if not texts:
            return []

        self._ensure_model_loaded()

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._generate_sync, texts)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise EmbeddingError(
                f"Failed to generate embeddings: {str(e)}",
                provider="local",
                error_type="generation",
            )

    def _generate_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings synchronously."""
        with self._torch.no_grad():
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=True,
                device=self._model.device,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            # Convert to list of lists for compatibility
            return embeddings.cpu().numpy().tolist()


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    @staticmethod
    def create_provider(
        provider_id: str, provider_config: EmbeddingProviderConfig
    ) -> EmbeddingProvider:
        """Create an embedding provider instance.

        Args:
            provider_id: Unique identifier for this provider
            provider_config: Provider configuration

        Returns:
            EmbeddingProvider instance

        Raises:
            ValueError: If provider type is unknown
        """
        if provider_config.provider_type == "openai":
            return OpenAIEmbeddingProvider(provider_id, provider_config)
        elif provider_config.provider_type == "local":
            return LocalEmbeddingProvider(provider_id, provider_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_config.provider_type}")

    @staticmethod
    def create_from_config(config_key: Optional[str] = None) -> EmbeddingProvider:
        """Create provider from configuration.

        Args:
            config_key: Optional provider key from config. If None, uses default.

        Returns:
            EmbeddingProvider instance
        """
        config = get_config()

        if config_key is None:
            config_key = config.embedding.default_provider

        if config_key not in config.embedding.providers:
            raise ConfigurationError(
                f"Provider '{config_key}' not found in configuration. "
                f"Available providers: {list(config.embedding.providers.keys())}"
            )

        provider_config = config.embedding.providers[config_key]
        return EmbeddingProviderFactory.create_provider(config_key, provider_config)
