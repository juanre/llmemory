# ABOUTME: Reranking utilities for refining search results using lexical heuristics or custom scorers.
# ABOUTME: Supports async scoring callbacks and configurable candidate limits for reranking.

"""Reranking utilities for post-retrieval optimization."""

import asyncio
import logging
import math
import re
from contextlib import suppress
from dataclasses import replace as dc_replace
from typing import Awaitable, Callable, Iterable, List, Optional, Sequence, Tuple

from .config import SearchConfig
from .models import SearchResult

logger = logging.getLogger(__name__)

ScoreCallback = Callable[[str, Sequence[SearchResult]], Awaitable[Sequence[float]]]

try:  # Optional heavy dependency
    from sentence_transformers import CrossEncoder as _SentenceCrossEncoder  # type: ignore
except Exception:  # pragma: no cover - handled gracefully when dependency missing
    _SentenceCrossEncoder = None  # type: ignore

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - handled gracefully
    AsyncOpenAI = None  # type: ignore


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
                    await asyncio.wait_for(self.score_callback(query_text, candidates), timeout=8.0)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Custom reranker callback failed: %s", exc)
                rerank_scores = []

        if not rerank_scores or len(rerank_scores) != len(candidates):
            rerank_scores = self._lexical_scores(query_text, candidates)

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
            token_overlap += sum(1 for token in query_tokens if token in metadata_tokens)

        length_penalty = math.log(len(content_tokens) + 1)
        return self.keyword_boost * token_overlap / length_penalty

    def _lexical_scores(self, query_text: str, results: Sequence[SearchResult]) -> List[float]:
        return [self._lexical_score(query_text, result) for result in results]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


class CrossEncoderReranker:
    """Cross-encoder reranker built on top of sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        if _SentenceCrossEncoder is None:
            raise ImportError(
                "sentence-transformers with cross-encoder support is required. "
                "Install optional dependencies via: uv add 'llmemory[reranker-local]'"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        with suppress(TypeError):
            self._model = _SentenceCrossEncoder(model_name, device=device)
        if not hasattr(self, "_model"):
            self._model = _SentenceCrossEncoder(model_name)

    async def score(self, query_text: str, results: Sequence[SearchResult]) -> Sequence[float]:
        if not results:
            return []

        loop = asyncio.get_running_loop()
        pairs = [(query_text, (res.summary or res.content or "")) for res in results]
        scores = await loop.run_in_executor(None, self._predict_sync, pairs)
        return scores

    def _predict_sync(self, pairs: List[Tuple[str, str]]) -> List[float]:
        predictions = self._model.predict(pairs, batch_size=self.batch_size)
        if hasattr(predictions, "tolist"):
            return predictions.tolist()  # type: ignore[return-value]
        return list(predictions)


class OpenAIResponsesReranker:
    """Reranker backed by the OpenAI responses API."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        max_candidates: int = 30,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai>=1.0.0 is required for provider='openai'.")

        self.model = model
        self.max_candidates = max_candidates
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key)

    async def score(self, query_text: str, results: Sequence[SearchResult]) -> Sequence[float]:
        if not results:
            return []

        limited_results = results[: self.max_candidates]
        prompt = self._build_prompt(query_text, limited_results)

        schema = {
            "name": "rerank_response",
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {"type": "number"},
                    }
                },
                "required": ["scores"],
                "additionalProperties": False,
            },
        }

        response = await self._client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation assistant. Given a query and passages, "
                        "return numeric relevance scores between 0 and 1 for each passage."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
        )

        scores = self._extract_scores(response)
        if len(scores) != len(limited_results):
            logger.warning(
                "OpenAI reranker returned %d scores for %d passages.",
                len(scores),
                len(limited_results),
            )
            return [0.0] * len(results)

        if len(results) > len(limited_results):
            scores.extend([0.0] * (len(results) - len(limited_results)))

        return scores

    def _build_prompt(self, query_text: str, results: Sequence[SearchResult]) -> str:
        lines = [f"Query: {query_text}", "", "Passages:"]
        for idx, res in enumerate(results, start=1):
            snippet = res.summary or res.content
            snippet = " ".join(snippet.split())[:400]
            lines.append(f"{idx}. {snippet}")

        lines.append(
            "\nReturn a JSON object with field `scores`, an array of floats between 0 and 1, "
            "matching the order of the passages."
        )
        return "\n".join(lines)

    def _extract_scores(self, response) -> List[float]:
        import json

        try:
            content = response.output[0].content[0].text  # type: ignore[attr-defined]
            payload = json.loads(content)
            return [float(x) for x in payload.get("scores", [])]
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to parse OpenAI reranker output: %s", exc)
            return []
