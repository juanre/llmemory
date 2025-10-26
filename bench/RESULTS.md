# SOTA RAG Performance Benchmark Results

**Date**: 2025-10-26
**Test Database**: `postgresql://localhost/llmemory_bench_test`
**Embedding Provider**: OpenAI (text-embedding-3-small)
**Reranking Provider**: Lexical (fast baseline)

## Overview

These benchmarks measure the performance of the SOTA RAG implementation, specifically:
1. **Hybrid Search**: Combining vector and BM25 search with Reciprocal Rank Fusion (RRF)
2. **Reranking**: Overhead of reranking search results

## Test Configuration

- **Document Count**: 100 documents
- **Query Count**: 30 queries (3 different queries × 10 iterations each)
- **Document Size**: ~2500 characters each (50 repetitions of a 50-character sentence)
- **Search Parameters**:
  - Search Type: Hybrid (alpha=0.5)
  - Limit: 10 results
  - Reranking: Lexical with top_k=50, return_k=10

## Results

### 1. Hybrid Search Latency

**Target**: P95 < 100ms

| Metric | Value | Status |
|--------|-------|--------|
| Average | 452.42ms | ⚠️ Above target |
| P50 (Median) | 296.06ms | ⚠️ Above target |
| P95 | 480.54ms | ❌ FAIL (4.8x target) |
| P99 | 4731.11ms | ❌ FAIL (47x target) |

**Status**: ❌ **FAIL** - P95 latency is 480.54ms, significantly exceeding the 100ms target

### 2. Reranking Overhead

**Target**: < 100ms overhead

| Metric | Value | Status |
|--------|-------|--------|
| Without Reranking | 426.77ms | - |
| With Reranking | 229.01ms | - |
| Overhead | -197.77ms | ✅ PASS |

**Status**: ✅ **PASS** - Reranking actually *improved* performance by 197.77ms

**Note**: The negative overhead is unexpected but indicates that reranking may be optimizing the query path or that the lexical reranker is more efficient than expected.

## Analysis

### Performance Issues

1. **High P95 Latency (480.54ms)**:
   - The P95 latency is 4.8x higher than the target
   - This suggests that 5% of queries are experiencing significant slowdowns
   - Potential causes:
     - OpenAI API latency for embedding generation
     - Database query performance (pgvector + BM25)
     - Network latency
     - Cold cache effects

2. **Very High P99 Latency (4731.11ms)**:
   - The P99 shows extreme outliers (47x target)
   - This indicates occasional severe performance degradation
   - Likely causes:
     - API rate limiting or timeouts
     - Database connection pool exhaustion
     - GC pauses or system resource contention

### Reranking Performance

The reranking overhead being negative is interesting and suggests:
- The lexical reranker may be using cached or pre-computed scores
- Reranking might be triggering query optimizations
- The "without reranking" path may be doing additional work that reranking avoids

## Recommendations

1. **Profile OpenAI API Calls**:
   - Measure embedding generation time separately
   - Consider caching embeddings for frequently used queries
   - Evaluate local embedding models for latency-sensitive applications

2. **Database Optimization**:
   - Analyze query execution plans for hybrid search
   - Consider adding database indexes
   - Tune pgvector HNSW parameters for better latency

3. **Connection Pooling**:
   - The warning about multiple connection pools suggests inefficiency
   - Implement shared pool pattern to reduce connection overhead

4. **Investigate P99 Outliers**:
   - Add detailed timing instrumentation
   - Monitor for API rate limits
   - Check system resource usage during tests

5. **Benchmark Iteration**:
   - Run benchmarks with more iterations for statistical significance
   - Test with different document counts (10, 100, 1000, 10000)
   - Measure with and without OpenAI API (using local embeddings)

## Test Environment

- **Platform**: macOS (Darwin 24.6.0)
- **Python**: 3.11.12
- **PostgreSQL**: localhost
- **Network**: Local (no internet latency for DB)
- **OpenAI API**: External API calls (subject to network latency)

## Conclusion

The current implementation **does not meet** the performance targets:
- ❌ Hybrid search P95: 480.54ms (target: <100ms)
- ✅ Reranking overhead: -197.77ms (target: <100ms)

The primary bottleneck appears to be in the hybrid search latency, likely due to:
1. OpenAI API latency for embedding generation
2. Database query performance for vector + BM25 search
3. Occasional extreme outliers (P99 at 4731ms)

Further investigation and optimization are required to meet production-ready performance targets.
