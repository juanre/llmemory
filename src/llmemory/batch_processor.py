"""Batch processing for embeddings with rate limiting and retry logic.

Based on agent-engine integration requirements:
- Process up to 100 docs/minute throughput
- Respect OpenAI rate limits (3000 RPM)
- Implement retry logic with exponential backoff
- Track batch progress
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingGenerator
from .models import DocumentChunk, EmbeddingJob, EmbeddingStatus

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API calls."""

    max_rpm: int = 3000  # Max requests per minute
    window_size: int = 60  # Window size in seconds

    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        self._tokens = float(self.max_rpm)
        self._last_update = time.time()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary."""
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self._last_update

                # Refill tokens based on elapsed time
                tokens_to_add = elapsed * (self.max_rpm / self.window_size)
                self._tokens = min(self.max_rpm, self._tokens + tokens_to_add)
                self._last_update = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / (self.max_rpm / self.window_size)

                logger.debug(
                    f"Rate limit: waiting {wait_time:.2f}s for {tokens} tokens"
                )
                await asyncio.sleep(wait_time)


@dataclass
class BatchProgress:
    """Track batch processing progress."""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return self.processed_chunks + self.failed_chunks >= self.total_chunks

    @property
    def success_rate(self) -> float:
        if self.processed_chunks + self.failed_chunks == 0:
            return 0.0
        return self.processed_chunks / (self.processed_chunks + self.failed_chunks)

    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def throughput(self) -> float:
        """Documents per minute."""
        minutes = self.duration.total_seconds() / 60
        if minutes == 0:
            return 0.0
        return self.processed_chunks / minutes


class BatchEmbeddingProcessor:
    """Process embeddings in batches with rate limiting and retry logic."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        batch_size: int = 100,
        max_rpm: int = 3000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize batch processor.

        Args:
            embedding_generator: Generator for creating embeddings
            batch_size: Maximum chunks per batch
            max_rpm: Maximum requests per minute
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
        """
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = RateLimiter(max_rpm=max_rpm)
        self.progress = BatchProgress()

    async def process_chunks(
        self, chunks: List[DocumentChunk], update_callback: Optional[callable] = None
    ) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """
        Process chunks in batches.

        Args:
            chunks: List of chunks to process
            update_callback: Optional callback for progress updates

        Returns:
            Tuple of (successful_chunks, failed_chunks_with_errors)
        """
        self.progress = BatchProgress(total_chunks=len(chunks))

        successful_chunks = []
        failed_chunks = []

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]

            # Rate limit check
            await self.rate_limiter.acquire(len(batch))

            # Process batch with retry logic
            success, failures = await self._process_batch_with_retry(batch)

            successful_chunks.extend(success)
            failed_chunks.extend(failures)

            # Update progress
            self.progress.processed_chunks += len(success)
            self.progress.failed_chunks += len(failures)

            # Callback for progress updates
            if update_callback:
                await update_callback(self.progress)

            # Log progress
            logger.info(
                f"Batch progress: {self.progress.processed_chunks}/{self.progress.total_chunks} "
                f"({self.progress.success_rate:.1%} success rate, "
                f"{self.progress.throughput:.1f} docs/min)"
            )

        self.progress.end_time = datetime.now()

        logger.info(
            f"Batch processing complete: {self.progress.processed_chunks} successful, "
            f"{self.progress.failed_chunks} failed in {self.progress.duration}"
        )

        return successful_chunks, failed_chunks

    async def _process_batch_with_retry(
        self, batch: List[DocumentChunk]
    ) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """
        Process a single batch with retry logic.

        Args:
            batch: Batch of chunks to process

        Returns:
            Tuple of (successful_chunks, failed_chunks_with_errors)
        """
        texts = [chunk.content for chunk in batch]
        retry_count = 0
        delay = self.retry_delay

        while retry_count <= self.max_retries:
            try:
                # Generate embeddings
                embeddings = await self.embedding_generator.generate_embeddings(texts)

                # Update chunks with embeddings
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding

                return batch, []

            except Exception as e:
                retry_count += 1

                if retry_count > self.max_retries:
                    logger.error(f"Batch failed after {self.max_retries} retries: {e}")

                    # Return all as failed
                    failed = [
                        {"chunk": chunk, "error": str(e), "retry_count": retry_count}
                        for chunk in batch
                    ]
                    return [], failed

                # Exponential backoff
                logger.warning(
                    f"Batch failed, retry {retry_count}/{self.max_retries} in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        return [], []

    async def process_embedding_jobs(
        self,
        jobs: List[EmbeddingJob],
        get_chunk_content: callable,
        update_chunk_embedding: callable,
        update_job_status: callable,
    ) -> BatchProgress:
        """
        Process embedding jobs from the queue.

        Args:
            jobs: List of embedding jobs to process
            get_chunk_content: Async function to get chunk content by ID
            update_chunk_embedding: Async function to update chunk embedding
            update_job_status: Async function to update job status

        Returns:
            BatchProgress with results
        """
        self.progress = BatchProgress(total_chunks=len(jobs))

        # Group jobs into batches
        for i in range(0, len(jobs), self.batch_size):
            batch_jobs = jobs[i : i + self.batch_size]

            # Get chunk contents
            chunk_contents = []
            for job in batch_jobs:
                try:
                    content = await get_chunk_content(job.chunk_id)
                    chunk_contents.append((job, content))
                except Exception as e:
                    logger.error(f"Failed to get content for chunk {job.chunk_id}: {e}")
                    await update_job_status(
                        job.queue_id, EmbeddingStatus.FAILED, error_message=str(e)
                    )
                    self.progress.failed_chunks += 1
                    continue

            if not chunk_contents:
                continue

            # Rate limit
            await self.rate_limiter.acquire(len(chunk_contents))

            # Generate embeddings
            texts = [content for _, content in chunk_contents]

            try:
                embeddings = await self.embedding_generator.generate_embeddings(texts)

                # Update chunks and jobs
                for (job, _), embedding in zip(chunk_contents, embeddings):
                    try:
                        await update_chunk_embedding(job.chunk_id, embedding)
                        await update_job_status(job.queue_id, EmbeddingStatus.COMPLETED)
                        self.progress.processed_chunks += 1

                    except Exception as e:
                        logger.error(f"Failed to update chunk {job.chunk_id}: {e}")
                        await update_job_status(
                            job.queue_id, EmbeddingStatus.FAILED, error_message=str(e)
                        )
                        self.progress.failed_chunks += 1

            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")

                # Mark all jobs in batch as failed
                for job, _ in chunk_contents:
                    await update_job_status(
                        job.queue_id, EmbeddingStatus.FAILED, error_message=str(e)
                    )
                    self.progress.failed_chunks += 1

            # Log progress
            logger.info(
                f"Job processing progress: {self.progress.processed_chunks}/{self.progress.total_chunks} "
                f"({self.progress.throughput:.1f} docs/min)"
            )

        self.progress.end_time = datetime.now()
        return self.progress


class BackgroundEmbeddingProcessor:
    """Background agent for continuous embedding processing."""

    def __init__(
        self,
        batch_processor: BatchEmbeddingProcessor,
        poll_interval: float = 30.0,
        batch_timeout: float = 60.0,
    ):
        """
        Initialize background processor.

        Args:
            batch_processor: Batch processor instance
            poll_interval: Seconds between queue polls
            batch_timeout: Maximum seconds to wait for full batch
        """
        self.batch_processor = batch_processor
        self.poll_interval = poll_interval
        self.batch_timeout = batch_timeout
        self._running = False
        self._task: Optional[asyncio.Agent] = None

    async def start(
        self,
        get_pending_jobs: callable,
        get_chunk_content: callable,
        update_chunk_embedding: callable,
        update_job_status: callable,
    ):
        """Start background processing."""
        if self._running:
            logger.warning("Background processor already running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._process_loop(
                get_pending_jobs,
                get_chunk_content,
                update_chunk_embedding,
                update_job_status,
            )
        )
        logger.info("Background embedding processor started")

    async def stop(self):
        """Stop background processing."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Background embedding processor stopped")

    async def _process_loop(
        self,
        get_pending_jobs: callable,
        get_chunk_content: callable,
        update_chunk_embedding: callable,
        update_job_status: callable,
    ):
        """Main processing loop."""
        while self._running:
            try:
                # Get pending jobs
                jobs = await get_pending_jobs(
                    limit=self.batch_processor.batch_size * 10
                )

                if jobs:
                    logger.info(f"Processing {len(jobs)} pending embedding jobs")

                    # Process jobs
                    progress = await self.batch_processor.process_embedding_jobs(
                        jobs,
                        get_chunk_content,
                        update_chunk_embedding,
                        update_job_status,
                    )

                    logger.info(
                        f"Processed batch: {progress.processed_chunks} successful, "
                        f"{progress.failed_chunks} failed, "
                        f"throughput: {progress.throughput:.1f} docs/min"
                    )

                    # Short delay if we processed jobs
                    await asyncio.sleep(1)
                else:
                    # No jobs, wait longer
                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in background processing loop: {e}")
                await asyncio.sleep(self.poll_interval)
