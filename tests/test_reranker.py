from typing import List
from uuid import uuid4

import pytest

from llmemory.config import SearchConfig
from llmemory.models import SearchResult
from llmemory.reranker import CrossEncoderReranker, OpenAIResponsesReranker, RerankerService


@pytest.mark.asyncio
async def test_cross_encoder_reranker_scores():
    pytest.importorskip("sentence_transformers")

    try:
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            batch_size=2,
        )
    except OSError as exc:
        pytest.skip(f"Model download unavailable: {exc}")

    results: List[SearchResult] = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="Python makes data analysis straightforward with pandas and NumPy.",
            metadata={},
            score=0.0,
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="Neural networks are powerful for image recognition tasks.",
            metadata={},
            score=0.0,
        ),
    ]

    scores = await reranker.score("python data analysis", results)
    assert len(scores) == len(results)
    assert scores[0] != scores[1]


@pytest.mark.asyncio
async def test_reranker_service_uses_callback():
    async def scorer(query: str, _results: List[SearchResult]):
        return [len(query) for _ in _results]

    service = RerankerService(SearchConfig(), score_callback=scorer)
    hits: List[SearchResult] = []
    for text in ["foo", "bar", "baz"]:
        sr = SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content=text,
            metadata={},
            score=0.0,
        )
        hits.append(sr)

    reranked = await service.rerank("abcde", hits, top_k=3, return_k=2)
    assert len(reranked) == len(hits)
    # Scores should reflect callback output
    assert reranked[0].score >= reranked[1].score


@pytest.mark.asyncio
async def test_openai_reranker(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.output = [
                type(
                    "Item",
                    (),
                    {"content": [type("Piece", (), {"text": '{"scores":[0.7,0.2]}'})]},
                )
            ]

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.responses = self

        async def create(self, **kwargs):
            return DummyResponse()

    monkeypatch.setattr("llmemory.reranker.AsyncOpenAI", DummyClient)

    reranker = OpenAIResponsesReranker(model="fake-model", max_candidates=5)
    results = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="passage one",
            metadata={},
            score=0.0,
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="passage two",
            metadata={},
            score=0.0,
        ),
    ]

    scores = await reranker.score("sample query", results)
    assert scores[:2] == [0.7, 0.2]
