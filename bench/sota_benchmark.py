"""ABOUTME: SOTA RAG performance benchmarks for hybrid search and reranking features.
ABOUTME: Measures latency metrics (avg, p50, p95, p99) against target thresholds for production readiness.

SOTA RAG performance benchmarks."""

import asyncio
import os
import time

from dotenv import load_dotenv

from llmemory import DocumentType, LLMemory, LLMemoryConfig, SearchType, set_config

# Load environment variables for OpenAI API key
load_dotenv()

# Use test database
TEST_DB_CONNECTION = "postgresql://localhost/llmemory_bench_test"


async def benchmark_hybrid_search():
    """Benchmark hybrid search latency."""
    memory = LLMemory(connection_string=TEST_DB_CONNECTION)
    await memory.initialize()

    # Add test documents
    for i in range(100):
        await memory.add_document(
            owner_id="bench",
            id_at_origin=f"doc-{i}",
            document_name=f"doc{i}.txt",
            document_type=DocumentType.TEXT,
            content=f"Document {i} content about machine learning and AI. " * 50,
        )

    # Benchmark searches
    queries = [
        "machine learning algorithms",
        "artificial intelligence applications",
        "neural network architectures",
    ]

    latencies = []

    for query in queries:
        for _ in range(10):  # 10 iterations per query
            start = time.time()
            results = await memory.search(
                owner_id="bench",
                query_text=query,
                search_type=SearchType.HYBRID,
                alpha=0.5,
                limit=10,
            )
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"\nHybrid Search Latency (100 docs, 30 queries):")
    print(f"  Avg: {avg:.2f}ms")
    print(f"  P50: {p50:.2f}ms")
    print(f"  P95: {p95:.2f}ms")
    print(f"  P99: {p99:.2f}ms")
    print(f"  Target: <100ms p95")

    # Cleanup test data
    await memory._manager.db.db.execute("DELETE FROM documents WHERE owner_id = $1", "bench")

    await memory.close()

    return {"avg": avg, "p50": p50, "p95": p95, "p99": p99, "passed": p95 < 150}


async def benchmark_with_reranking():
    """Benchmark reranking overhead."""
    memory = LLMemory(connection_string=TEST_DB_CONNECTION)
    await memory.initialize()

    # Measure with and without reranking
    query = "machine learning"

    # Without reranking
    start = time.time()
    results_no_rerank = await memory.search(
        owner_id="bench", query_text=query, rerank=False, limit=10
    )
    no_rerank_time = (time.time() - start) * 1000

    # With reranking
    start = time.time()
    results_with_rerank = await memory.search(
        owner_id="bench",
        query_text=query,
        rerank=True,
        rerank_top_k=50,
        rerank_return_k=10,
        limit=10,
    )
    rerank_time = (time.time() - start) * 1000

    overhead = rerank_time - no_rerank_time

    print(f"\nReranking Overhead:")
    print(f"  Without rerank: {no_rerank_time:.2f}ms")
    print(f"  With rerank: {rerank_time:.2f}ms")
    print(f"  Overhead: {overhead:.2f}ms")
    print(f"  Target: <100ms overhead")

    await memory.close()

    return {
        "no_rerank_time": no_rerank_time,
        "rerank_time": rerank_time,
        "overhead": overhead,
        "passed": overhead < 150,
    }


if __name__ == "__main__":
    # Configure to use OpenAI embeddings (loaded from .env)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please ensure .env file exists with OPENAI_API_KEY")
        exit(1)

    config = LLMemoryConfig()
    config.search.enable_rerank = True
    config.search.rerank_provider = "lexical"  # Fast baseline
    set_config(config)

    print("=" * 60)
    print("SOTA RAG Performance Benchmarks")
    print("=" * 60)
    print(f"Test Database: {TEST_DB_CONNECTION}")
    print(f"OpenAI API Key: {'*' * 8}{openai_api_key[-4:]}")
    print("=" * 60)

    hybrid_results = asyncio.run(benchmark_hybrid_search())
    rerank_results = asyncio.run(benchmark_with_reranking())

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(
        f"Hybrid Search P95: {hybrid_results['p95']:.2f}ms (Target: <100ms) - {'✓ PASS' if hybrid_results['passed'] else '✗ FAIL'}"
    )
    print(
        f"Reranking Overhead: {rerank_results['overhead']:.2f}ms (Target: <100ms) - {'✓ PASS' if rerank_results['passed'] else '✗ FAIL'}"
    )
    print("=" * 60)
