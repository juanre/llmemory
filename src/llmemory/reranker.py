# ABOUTME: Reranking utilities for refining search results using lexical heuristics or custom scorers.
# ABOUTME: Supports async scoring callbacks and configurable candidate limits for reranking.

"""Reranking utilities for post-retrieval optimization."""

import asyncio
import logging
import math
import re
from dataclasses import replace as dc_replace
from typing import Awaitable, Callable, Iterable, List, Optional, Sequence, Tuple

from .config import SearchConfig
from .models import SearchResult

logger = logging.getLogger(__name__)

ScoreCallback = Callable[[str, Sequence[SearchResult]], Awaitable[Sequence[float]]]


class RerankerService:
    """Apply reranking to search results."""

    def __init__(
        self,
        search_config: SearchConfig,
        score_callback: Optional[ScoreCallback] = None,
        keyword_boost: float = 1.0,
    ) -> None:
        self.config = search_config
        self.score_callback = score_callback
        self.keyword_boost = keyword_boost

    async def rerank(
        self,
        query_text: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        return_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank results and return a reordered list."""
        if not results:
            return results

        candidate_count = top_k or self.config.rerank_top_k
        candidate_count = max(1, candidate_count)
        candidates = results[: min(candidate_count, len(results))]

        rerank_scores = []
        if self.score_callback:
            try:
                rerank_scores = list(
                    await asyncio.wait_for(
                        self.score_callback(query_text, candidates), timeout=8.0
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Custom reranker callback failed: %s", exc)
                rerank_scores = []

        if not rerank_scores or len(rerank_scores) != len(candidates):
            rerank_scores = [
                self._lexical_score(query_text, result) for result in candidates
            ]

        scored_candidates: List[Tuple[float, SearchResult]] = []
        for original_score, candidate in zip(rerank_scores, candidates):
            rerank_score = float(original_score)
            # Incorporate prior score as a small tiebreaker
            tiebreaker = candidate.rrf_score or candidate.score or 0.0
            combined = rerank_score + 0.001 * tiebreaker
            scored_candidates.append((combined, candidate))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        desired = return_k or self.config.rerank_return_k
        desired = max(1, desired)

        reranked: List[SearchResult] = []
        for combined_score, candidate in scored_candidates:
            reranked.append(
                dc_replace(
                    candidate,
                    score=combined_score,
                    rerank_score=combined_score,
                )
            )
            if len(reranked) >= desired:
                break

        # Append any remaining results to fulfill caller's limit, preserving order
        if len(reranked) < len(results):
            existing_ids = {res.chunk_id for res in reranked}
            for remaining in results:
                if remaining.chunk_id in existing_ids:
                    continue
                reranked.append(remaining)

        return reranked

    def _lexical_score(self, query_text: str, result: SearchResult) -> float:
        """Simple lexical scoring based on token overlap."""
        if not query_text:
            return 0.0

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return 0.0

        content_tokens = self._tokenize(result.content)
        if not content_tokens:
            return 0.0

        token_overlap = sum(1 for token in query_tokens if token in content_tokens)
        if result.metadata:
            metadata_text = " ".join(f"{k} {v}" for k, v in result.metadata.items())
            metadata_tokens = self._tokenize(metadata_text)
            token_overlap += sum(
                1 for token in query_tokens if token in metadata_tokens
            )

        length_penalty = math.log(len(content_tokens) + 1)
        return self.keyword_boost * token_overlap / length_penalty

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
