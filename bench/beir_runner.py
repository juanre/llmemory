"""Benchmark harness for evaluating llmemory on BEIR datasets."""

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from llmemory import LLMemory, DocumentType, SearchType

logger = logging.getLogger(__name__)


async def ingest_corpus(
    memory: LLMemory,
    owner_id: str,
    corpus: Dict[str, Dict[str, str]],
    chunking_strategy: str,
    limit: int = 0,
) -> Dict[str, str]:
    doc_id_map: Dict[str, str] = {}
    start = time.time()

    for idx, (doc_id, doc) in enumerate(corpus.items(), start=1):
        if limit and idx > limit:
            break

        content = "\n\n".join(filter(None, [doc.get("title", ""), doc.get("text", "")]))
        if not content:
            continue

        result = await memory.add_document(
            owner_id=owner_id,
            id_at_origin=doc_id,
            document_name=f"{doc_id}.txt",
            document_type=DocumentType.TEXT,
            content=content,
            metadata={"beir_doc_id": doc_id},
            chunking_strategy=chunking_strategy,
            generate_embeddings=True,
        )
        doc_id_map[str(result.document.document_id)] = doc_id

        if idx % 1000 == 0:
            logger.info("Ingested %d documents", idx)

    logger.info("Ingestion completed in %.2fs", time.time() - start)
    return doc_id_map


async def run_queries(
    memory: LLMemory,
    owner_id: str,
    queries: Dict[str, str],
    doc_id_map: Dict[str, str],
    top_k: int,
    search_type: SearchType,
    use_query_expansion: bool,
    use_rerank: bool,
    limit: int = 0,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    start = time.time()

    for idx, (query_id, query_text) in enumerate(queries.items(), start=1):
        if limit and idx > limit:
            break

        search_results = await memory.search(
            owner_id=owner_id,
            query_text=query_text,
            search_type=search_type,
            limit=top_k,
            query_expansion=use_query_expansion,
            rerank=use_rerank,
        )

        ranked: Dict[str, float] = {}
        for res in search_results:
            beir_id = doc_id_map.get(str(res.document_id))
            if not beir_id or beir_id in ranked:
                continue
            ranked[beir_id] = float(res.score)

        results[query_id] = ranked

        if idx % 100 == 0:
            logger.info("Executed %d queries", idx)

    logger.info("Query phase completed in %.2fs", time.time() - start)
    return results


def evaluate(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    if not run:
        return {}

    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, run, [1, 3, 5, 10, 20, 100])
    return {metric: values.get("all", 0.0) for metric, values in metrics.items()}


async def benchmark(args) -> None:
    data_path = Path(args.dataset_dir).expanduser()
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(data_path / args.dataset)
    ).load(split=args.split)

    memory = LLMemory(connection_string=args.connection)
    await memory.initialize()

    try:
        doc_map = await ingest_corpus(
            memory,
            owner_id=args.owner_id,
            corpus=corpus,
            chunking_strategy=args.chunking_strategy,
            limit=args.doc_limit,
        )

        run = await run_queries(
            memory,
            owner_id=args.owner_id,
            queries=queries,
            doc_id_map=doc_map,
            top_k=args.top_k,
            search_type=SearchType(args.search_type),
            use_query_expansion=args.query_expansion,
            use_rerank=args.rerank,
            limit=args.query_limit,
        )

        metrics = evaluate(run, qrels)
        for metric, value in metrics.items():
            logger.info("%s: %.4f", metric, value)
    finally:
        await memory.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BEIR benchmark against llmemory")
    parser.add_argument("dataset", help="BEIR dataset name (e.g., nq, fiqa, hotpotqa)")
    parser.add_argument(
        "--dataset-dir",
        default="datasets",
        help="Directory containing downloaded BEIR datasets",
    )
    parser.add_argument("--connection", default="postgresql://localhost/llmemory_bench")
    parser.add_argument("--owner-id", default="beir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--search-type", default="hybrid", choices=[t.value for t in SearchType])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-expansion", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--chunking-strategy",
        default="hierarchical",
        help="Chunking strategy used during ingestion",
    )
    parser.add_argument("--doc-limit", type=int, default=0, help="Limit number of documents ingested")
    parser.add_argument("--query-limit", type=int, default=0, help="Limit number of queries executed")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
