# ABOUTME: Query expansion utilities generating synthetic variants for multi-query retrieval.
# ABOUTME: Provides configurable heuristic expansion with optional LLM callbacks and deduplication.

"""Utilities for generating synthetic query variants."""

import asyncio
import logging
import re
from typing import Awaitable, Callable, Iterable, List, Optional, Sequence, Set

from .config import SearchConfig

logger = logging.getLogger(__name__)

ExpansionCallback = Callable[[str, int], Awaitable[Sequence[str]]]

# Basic stopword list for keyword variant generation
DEFAULT_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


class QueryExpansionService:
    """Generate synthetic query variants for improved recall."""

    def __init__(
        self,
        search_config: SearchConfig,
        llm_callback: Optional[ExpansionCallback] = None,
        stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = search_config
        self.llm_callback = llm_callback
        self.stopwords = set(stopwords or DEFAULT_STOPWORDS)

    async def expand(
        self,
        query_text: str,
        max_variants: Optional[int] = None,
        include_keyword_variant: Optional[bool] = None,
    ) -> List[str]:
        """Generate query variants excluding the original query.

        Args:
            query_text: Original user query.
            max_variants: Maximum number of variants to return.
            include_keyword_variant: Whether to include heuristic keyword variant.

        Returns:
            List of unique variant strings (original not included).
        """
        query_text = query_text.strip()
        if not query_text:
            return []

        limit = max_variants if max_variants is not None else self.config.max_query_variants
        if limit <= 0:
            return []

        include_keywords = (
            include_keyword_variant
            if include_keyword_variant is not None
            else self.config.include_keyword_variant
        )

        candidates: List[str] = []
        seen: Set[str] = {query_text.lower()}

        # Attempt LLM-based expansion first if provided
        if self.llm_callback:
            try:
                llm_variants = await asyncio.wait_for(
                    self.llm_callback(query_text, limit), timeout=8.0
                )
                for variant in llm_variants:
                    normalized = variant.strip()
                    if not normalized:
                        continue
                    lowered = normalized.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    candidates.append(normalized)
                    if len(candidates) >= limit:
                        return candidates
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM query expansion failed: %s", exc)

        # Fall back to heuristic variants
        heuristic_variants = self._heuristic_variants(query_text, include_keywords)
        for variant in heuristic_variants:
            lowered = variant.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            candidates.append(variant)
            if len(candidates) >= limit:
                break

        return candidates[:limit]

    def _heuristic_variants(self, query_text: str, include_keywords: bool) -> List[str]:
        """Generate deterministic heuristic variants."""
        variants: List[str] = []

        normalized = re.sub(r"\s+", " ", query_text).strip()
        lower = normalized.lower()

        if include_keywords:
            keyword_variant = self._keyword_variant(lower)
            if keyword_variant and keyword_variant != lower:
                variants.append(keyword_variant)

        # Provide an OR-based variant to widen lexical recall
        tokens = [t for t in re.findall(r"\w+", lower) if t]
        unique_tokens = []
        for token in tokens:
            if token not in self.stopwords and token not in unique_tokens:
                unique_tokens.append(token)

        if len(unique_tokens) > 1:
            variants.append(" OR ".join(unique_tokens))

        # Include a quoted phrase variant when multi-word query
        if len(tokens) > 1:
            variants.append(f"\"{normalized}\"")

        return variants

    def _keyword_variant(self, query_text: str) -> str:
        """Create keyword-only variant removing stop words."""
        tokens = [t for t in re.findall(r"\w+", query_text) if t]
        keywords = [t for t in tokens if t not in self.stopwords]

        if not keywords:
            keywords = tokens

        return " ".join(keywords)
