import asyncio

import pytest

pytest.importorskip("beir")

from bench.beir_runner import evaluate
from llmemory import SearchType


def test_evaluate_with_empty_run():
    metrics = evaluate({}, {})
    assert isinstance(metrics, dict)
    assert metrics == {}


def test_run_queries_mapping(monkeypatch):
    from bench import beir_runner

    async def fake_search(**kwargs):
        class Result:
            def __init__(self, doc_id, score):
                self.document_id = doc_id
                self.score = score

        return [Result("uuid1", 0.5), Result("uuid2", 0.3)]

    class FakeMemory:
        async def search(self, **kwargs):
            return await fake_search()

    memory = FakeMemory()
    doc_map = {"uuid1": "docA", "uuid2": "docB"}
    queries = {"q1": "test"}

    results = asyncio.run(
        beir_runner.run_queries(
            memory,
            owner_id="owner",
            queries=queries,
            doc_id_map=doc_map,
            top_k=10,
            search_type=SearchType.HYBRID,
            use_query_expansion=False,
            use_rerank=False,
        )
    )

    assert "q1" in results
    assert set(results["q1"].keys()) == {"docA", "docB"}
