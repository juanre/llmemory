# ABOUTME: Batch embedding generation coordinator managing concurrent requests with provider rate limits and retry logic.
# ABOUTME: Orchestrates embedding generation across chunks with intelligent batching, error handling, and progress tracking.

"""Batch embedding generation with rate limiting and retries."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .embedding_providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles batch embedding generation using embedding providers."""

    def __init__(self, provider: EmbeddingProvider):
        """
        Initialize embedding generator with a provider.

        Args:
            provider: EmbeddingProvider instance to use
        """
        self.provider = provider

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        return await self.provider.generate_embedding(text)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If rate limit is exceeded
        """
        if not texts:
            return []

        return await self.provider.generate_embeddings(texts)

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "provider_id": self.provider.provider_id,
            "model_name": self.provider.get_model_name(),
            "dimension": self.provider.get_dimension(),
            "max_tokens": self.provider.config.max_tokens_per_minute,
        }

    async def process_batch(
        self, chunks: List[Dict[str, Any]], update_callback: Optional[callable] = None
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Process a batch of chunks and generate embeddings.

        Args:
            chunks: List of chunks with 'chunk_id' and 'content'
            update_callback: Optional callback to report progress

        Returns:
            Tuple of (chunk_ids, embeddings)
        """
        if not chunks:
            return [], []

        # Extract texts and chunk IDs
        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings
        try:
            embeddings = await self.generate_embeddings(texts)

            if update_callback:
                await update_callback(len(embeddings), 0)

            return chunk_ids, embeddings

        except Exception:
            if update_callback:
                await update_callback(0, len(texts))
            raise
