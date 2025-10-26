# ABOUTME: Query routing for RAG systems determining if queries are answerable from available documents.
# ABOUTME: Uses LLM to classify queries and route to appropriate handlers (retrieval, web, unanswerable).

"""Query routing for RAG systems."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    """Query routing decision types."""

    RETRIEVAL = "retrieval"  # Answer from documents
    WEB_SEARCH = "web_search"  # Need external info
    UNANSWERABLE = "unanswerable"  # Cannot answer
    CLARIFICATION = "clarification"  # Need more context


@dataclass
class RouteDecision:
    """Query routing decision."""

    route_type: RouteType
    confidence: float  # 0-1
    reason: str
    suggested_response: Optional[str] = None


class QueryRouter:
    """Routes queries based on answerability from available documents."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = openai_api_key
        self.model = model

    async def route(
        self, query: str, document_context: List[str], threshold: float = 0.7
    ) -> RouteDecision:
        """Determine how to handle a query.

        Args:
            query: User query text
            document_context: Sample of available documents (for context)
            threshold: Confidence threshold for routing

        Returns:
            RouteDecision with route type and confidence
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)

        # Build context sample
        context_sample = "\n\n".join(document_context[:5])  # Max 5 samples

        prompt = f"""Analyze if the following query can be answered using only the provided document context.

Document Context:
{context_sample}

User Query: {query}

Classify the query into one of:
1. RETRIEVAL - Can be answered from the documents above
2. WEB_SEARCH - Requires external/current information not in documents
3. UNANSWERABLE - Cannot be answered (too vague, opinion-based, etc.)
4. CLARIFICATION - Query is unclear, need more information

Respond with JSON:
{{
  "route": "RETRIEVAL|WEB_SEARCH|UNANSWERABLE|CLARIFICATION",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}"""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query classification expert for RAG systems.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=5.0,
            )

            import json

            result = json.loads(response.choices[0].message.content)

            route_type = RouteType(result["route"].lower())
            confidence = float(result["confidence"])
            reason = result["reason"]

            return RouteDecision(route_type=route_type, confidence=confidence, reason=reason)

        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            # Default to retrieval on error
            return RouteDecision(
                route_type=RouteType.RETRIEVAL,
                confidence=0.5,
                reason=f"Routing failed: {e}, defaulting to retrieval",
            )
