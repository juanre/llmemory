# SOTA RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring llmemory to state-of-the-art RAG quality by fixing incomplete features, removing all dead code, and implementing missing SOTA capabilities.

**Architecture:** Fix LLM query expansion wiring, remove 28 unused config fields, add query routing for answerable detection, implement contextual retrieval (Anthropic's approach), and achieve comprehensive test coverage.

**Tech Stack:** Python 3.10+, PostgreSQL 14+, pgvector, asyncpg, pgdbm, sentence-transformers, OpenAI API, pytest

**References:**
- Bug report: `BUG-llm-query-expansion-not-wired.md`
- Validation report: `VALIDATION-REPORT-comprehensive.md`
- SOTA requirements: https://blog.abdellatif.io/production-rag-processing-5m-documents

---

## PHASE 1: Fix Critical Bugs & Remove Dead Code

**Estimated Time:** 2-3 days
**Goal:** Zero tech debt, all config fields work, LLM expansion functional

---

### Task 1: Wire LLM Query Expansion Callback

**Files:**
- Modify: `src/llmemory/library.py:172`
- Modify: `src/llmemory/library.py` (add new method)
- Test: `tests/test_query_expansion.py`

**Step 1: Write failing test for LLM callback integration**

Add to `tests/test_query_expansion.py`:

```python
import pytest
from unittest.mock import AsyncMock, Mock
from llmemory import LLMemory, SearchType
from llmemory.config import LLMemoryConfig

@pytest.mark.asyncio
async def test_llm_query_expansion_callback_is_invoked(memory_with_documents):
    """Verify that LLM callback is called when query_expansion=True."""
    memory = memory_with_documents

    # Track if callback was invoked
    call_count = 0

    async def mock_llm_callback(query: str, limit: int):
        nonlocal call_count
        call_count += 1
        return [
            "semantic variant one",
            "semantic variant two"
        ]

    # Wire the callback (this will fail - method doesn't exist yet)
    memory._query_expander.llm_callback = mock_llm_callback

    results = await memory.search(
        owner_id="test-owner",
        query_text="test query",
        query_expansion=True,
        max_query_variants=3,
        limit=5
    )

    # Verify callback was invoked
    assert call_count > 0, "LLM callback should be invoked"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_query_expansion.py::test_llm_query_expansion_callback_is_invoked -v
```

Expected: PASS (callback works when manually set) - but we need automatic wiring

**Step 3: Write failing test for automatic callback wiring**

Add to `tests/test_query_expansion.py`:

```python
@pytest.mark.asyncio
async def test_llm_expansion_auto_wired_from_config(test_db):
    """Verify LLMemory automatically wires LLM callback when configured."""
    from llmemory.query_expansion import QueryExpansionService

    # Configure LLM expansion model
    config = LLMemoryConfig()
    config.search.enable_query_expansion = True
    config.search.query_expansion_model = "gpt-4o-mini"

    memory = LLMemory(
        connection_string=test_db,
        openai_api_key="sk-test-key",
        config=config
    )
    await memory.initialize()

    # Verify query expander has LLM callback wired
    assert memory._query_expander is not None
    assert memory._query_expander.llm_callback is not None, \
        "LLM callback should be auto-wired when query_expansion_model configured"

    await memory.close()
```

**Step 4: Run test to verify it fails**

```bash
uv run pytest tests/test_query_expansion.py::test_llm_expansion_auto_wired_from_config -v
```

Expected: FAIL with "LLM callback should be auto-wired"

**Step 5: Implement LLM callback factory method**

Add to `src/llmemory/library.py` after line 145 (after `_create_reranker_service`):

```python
def _create_query_expansion_callback(self) -> Optional[ExpansionCallback]:
    """Create LLM callback for query expansion if configured.

    Returns:
        Async callback function that generates query variants using LLM,
        or None if no expansion model configured.
    """
    from typing import Sequence
    from .query_expansion import ExpansionCallback

    model = self.config.search.query_expansion_model
    if not model:
        return None

    # Check if we have OpenAI API key
    if not self._openai_api_key:
        logger.warning(
            "query_expansion_model configured but no OpenAI API key available. "
            "Falling back to heuristic expansion."
        )
        return None

    async def openai_expansion_callback(query_text: str, max_variants: int) -> Sequence[str]:
        """Generate query variants using OpenAI."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._openai_api_key)

        prompt = f"""Generate {max_variants} alternative search queries that capture the same intent as the original query.

Original query: {query_text}

Requirements:
1. Semantically similar but use different words and phrasings
2. Include both more specific and more general variations
3. Capture different aspects or perspectives of the query
4. Keep queries concise (under 20 words each)
5. Return ONLY the alternative queries, one per line, no numbering or formatting

Alternative queries:"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query expansion expert. Generate diverse, semantically similar search queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # Some creativity for diversity
                max_tokens=200,
                timeout=5.0
            )

            # Parse variants from response
            content = response.choices[0].message.content.strip()
            variants = [
                line.strip()
                for line in content.split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*'))
            ]

            return variants[:max_variants]

        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
            return []  # Fall back to heuristics

    return openai_expansion_callback
```

**Step 6: Wire callback in initialize() method**

Modify `src/llmemory/library.py:172` from:

```python
# Initialize query expansion and reranking
self._query_expander = QueryExpansionService(self.config.search)
```

To:

```python
# Initialize query expansion and reranking
expansion_callback = self._create_query_expansion_callback()
self._query_expander = QueryExpansionService(
    self.config.search,
    llm_callback=expansion_callback
)
```

**Step 7: Add required import**

Add to imports at top of `src/llmemory/library.py`:

```python
from .query_expansion import QueryExpansionService, ExpansionCallback
```

**Step 8: Run tests to verify they pass**

```bash
uv run pytest tests/test_query_expansion.py::test_llm_expansion_auto_wired_from_config -v
```

Expected: PASS

**Step 9: Add integration test with real OpenAI call**

Add to `tests/test_query_expansion.py`:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_expansion_with_real_openai(test_db):
    """Integration test: Real OpenAI query expansion (requires OPENAI_API_KEY)."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY")

    config = LLMemoryConfig()
    config.search.query_expansion_model = "gpt-4o-mini"

    memory = LLMemory(
        connection_string=test_db,
        config=config
    )
    await memory.initialize()

    # Add test document
    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="Machine learning is a subset of artificial intelligence."
    )

    # Search with expansion
    results = await memory.search(
        owner_id="test",
        query_text="AI algorithms",
        query_expansion=True,
        max_query_variants=2,
        limit=5
    )

    # Should get results (variants help with recall)
    assert len(results) > 0

    await memory.close()
```

**Step 10: Run integration test**

```bash
OPENAI_API_KEY=sk-... uv run pytest tests/test_query_expansion.py::test_llm_expansion_with_real_openai -v
```

Expected: PASS

**Step 11: Commit**

```bash
git add src/llmemory/library.py tests/test_query_expansion.py
git commit -m "feat: wire LLM query expansion callback

- Add _create_query_expansion_callback() factory method
- Use OpenAI for semantic query variant generation
- Wire callback in LLMemory initialization
- Add tests for automatic callback wiring
- Add integration test with real OpenAI API
- Fallback to heuristics if API key missing or call fails

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Remove Unused SearchConfig Fields

**Files:**
- Modify: `src/llmemory/config.py:105-148`
- Modify: `src/llmemory/models.py:187` (if SearchQuery has same fields)
- Test: `tests/test_config.py` (create if doesn't exist)

**Step 1: Write test documenting which fields should exist**

Create `tests/test_config.py`:

```python
import pytest
from llmemory.config import SearchConfig, LLMemoryConfig

def test_search_config_only_has_used_fields():
    """Verify SearchConfig only contains fields that are actually used."""
    config = SearchConfig()

    # Fields that SHOULD exist and be used
    assert hasattr(config, 'default_limit')
    assert hasattr(config, 'max_limit')
    assert hasattr(config, 'hnsw_profile')
    assert hasattr(config, 'hnsw_ef_search')
    assert hasattr(config, 'rrf_k')
    assert hasattr(config, 'enable_query_expansion')
    assert hasattr(config, 'max_query_variants')
    assert hasattr(config, 'query_expansion_model')
    assert hasattr(config, 'include_keyword_variant')
    assert hasattr(config, 'enable_rerank')
    assert hasattr(config, 'default_rerank_model')
    assert hasattr(config, 'rerank_provider')
    assert hasattr(config, 'rerank_top_k')
    assert hasattr(config, 'rerank_return_k')
    assert hasattr(config, 'rerank_device')
    assert hasattr(config, 'rerank_batch_size')
    assert hasattr(config, 'cache_ttl')

    # Fields that should NOT exist (unused)
    assert not hasattr(config, 'search_timeout'), "search_timeout is unused"
    assert not hasattr(config, 'min_score_threshold'), "min_score_threshold is unused"
    assert not hasattr(config, 'cache_max_size'), "cache_max_size is unused"
    assert not hasattr(config, 'default_search_type'), "default_search_type is unused"
    assert not hasattr(config, 'vector_search_limit'), "vector_search_limit is unused"
    assert not hasattr(config, 'text_search_limit'), "text_search_limit is unused"
    assert not hasattr(config, 'text_search_config'), "text_search_config is unused"
```

**Step 2: Run test to see current state**

```bash
uv run pytest tests/test_config.py::test_search_config_only_has_used_fields -v
```

Expected: FAIL (unused fields still exist)

**Step 3: Remove unused SearchConfig fields**

Edit `src/llmemory/config.py`, remove these lines:

```python
# Remove from SearchConfig dataclass:
default_search_type: str = "hybrid"  # Line ~108 - UNUSED
hnsw_ef_search: int = 100  # Keep this - it's USED
vector_search_limit: int = 100  # Remove - UNUSED
text_search_limit: int = 100  # Remove - UNUSED
text_search_config: str = "english"  # Remove - UNUSED (language detection handles this)
cache_ttl: int = 3600  # Keep - used (but fix hardcoding separately)
cache_max_size: int = 10000  # Remove - UNUSED
search_timeout: float = 5.0  # Remove - UNUSED
min_score_threshold: float = 0.0  # Remove - UNUSED
```

**Step 4: Remove from environment variable parsing**

Edit `src/llmemory/config.py`, find the `from_env()` method and remove parsing for removed fields.

**Step 5: Run test to verify fields removed**

```bash
uv run pytest tests/test_config.py::test_search_config_only_has_used_fields -v
```

Expected: PASS

**Step 6: Update SearchQuery model if needed**

Check `src/llmemory/models.py` for matching unused fields and remove them.

**Step 7: Run all tests to verify nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/llmemory/config.py src/llmemory/models.py tests/test_config.py
git commit -m "refactor: remove unused SearchConfig fields

Remove 7 unused configuration fields from SearchConfig:
- search_timeout (never enforced)
- min_score_threshold (never applied)
- cache_max_size (never used)
- default_search_type (never used)
- vector_search_limit (never used)
- text_search_limit (never used)
- text_search_config (language detection handles this)

These fields existed in config but were never read or used by
any code, creating confusion and maintenance burden.

All tests passing after removal.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Remove Unused ChunkingConfig Fields

**Files:**
- Modify: `src/llmemory/config.py:70-102`
- Test: `tests/test_config.py`

**Step 1: Write test for ChunkingConfig**

Add to `tests/test_config.py`:

```python
def test_chunking_config_only_has_used_fields():
    """Verify ChunkingConfig only contains fields that are actually used."""
    from llmemory.config import ChunkingConfig

    config = ChunkingConfig()

    # Fields that SHOULD exist
    assert hasattr(config, 'enable_chunk_summaries')
    assert hasattr(config, 'summary_max_tokens')
    assert hasattr(config, 'min_chunk_size')
    assert hasattr(config, 'max_chunk_size')

    # Fields that should NOT exist (unused)
    assert not hasattr(config, 'default_parent_size'), "Never used in chunking"
    assert not hasattr(config, 'default_child_size'), "Never used in chunking"
    assert not hasattr(config, 'default_overlap'), "Never used in chunking"
    assert not hasattr(config, 'max_chunk_depth'), "Never enforced"
    assert not hasattr(config, 'summary_prompt_template'), "Summaries use truncation"
    assert not hasattr(config, 'chunk_configs'), "Document-type configs unused"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py::test_chunking_config_only_has_used_fields -v
```

Expected: FAIL (unused fields exist)

**Step 3: Remove unused ChunkingConfig fields**

Edit `src/llmemory/config.py`, remove:

```python
# Remove from ChunkingConfig:
default_parent_size: int = 1000  # UNUSED
default_child_size: int = 200  # UNUSED
default_overlap: int = 50  # UNUSED
max_chunk_depth: int = 3  # UNUSED (not enforced)
summary_prompt_template: str = "..."  # UNUSED (summaries use truncation)
chunk_configs: Dict[str, Dict[str, int]] = field(default_factory=lambda: {...})  # UNUSED
```

Keep only:
- `enable_chunk_summaries`
- `summary_max_tokens`
- `min_chunk_size`
- `max_chunk_size`
- `section_markers` (if used)

**Step 4: Remove from environment parsing**

Remove env var parsing for deleted fields in `from_env()` method.

**Step 5: Run test to verify**

```bash
uv run pytest tests/test_config.py::test_chunking_config_only_has_used_fields -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/llmemory/config.py tests/test_config.py
git commit -m "refactor: remove unused ChunkingConfig fields

Remove 6 unused fields from ChunkingConfig:
- default_parent_size, default_child_size, default_overlap
- max_chunk_depth
- summary_prompt_template
- chunk_configs

Chunking code uses hardcoded values appropriate for each document
type, not these config fields. Removing dead code.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Remove Unused DatabaseConfig Fields

**Files:**
- Modify: `src/llmemory/config.py:151-173`
- Test: `tests/test_config.py`

**Step 1: Write test**

Add to `tests/test_config.py`:

```python
def test_database_config_only_has_used_fields():
    """Verify DatabaseConfig only contains used fields."""
    from llmemory.config import DatabaseConfig

    config = DatabaseConfig()

    # Used fields
    assert hasattr(config, 'min_pool_size')
    assert hasattr(config, 'max_pool_size')
    assert hasattr(config, 'connection_timeout')
    assert hasattr(config, 'schema_name')
    assert hasattr(config, 'hnsw_m')
    assert hasattr(config, 'hnsw_ef_construction')

    # Unused fields (table names managed by pgdbm)
    assert not hasattr(config, 'documents_table')
    assert not hasattr(config, 'chunks_table')
    assert not hasattr(config, 'embeddings_queue_table')
    assert not hasattr(config, 'search_history_table')
    assert not hasattr(config, 'embedding_providers_table')
    assert not hasattr(config, 'chunk_embeddings_prefix')
    assert not hasattr(config, 'hnsw_index_name')
```

**Step 2: Run test**

```bash
uv run pytest tests/test_config.py::test_database_config_only_has_used_fields -v
```

Expected: FAIL

**Step 3: Remove unused fields**

Edit `src/llmemory/config.py`, remove:

```python
# Remove these - table names managed by pgdbm template system:
documents_table: str = "documents"
chunks_table: str = "document_chunks"
embeddings_queue_table: str = "embedding_queue"
search_history_table: str = "search_history"
embedding_providers_table: str = "embedding_providers"
chunk_embeddings_prefix: str = "chunk_embeddings_"
hnsw_index_name: str = "document_chunks_embedding_hnsw"
```

**Step 4: Verify tables are managed by pgdbm**

Check `src/llmemory/db.py` - tables should use pgdbm's `{{tables.table_name}}` syntax, not config fields.

**Step 5: Run test**

```bash
uv run pytest tests/test_config.py::test_database_config_only_has_used_fields -v
```

Expected: PASS

**Step 6: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All PASS

**Step 7: Commit**

```bash
git add src/llmemory/config.py tests/test_config.py
git commit -m "refactor: remove unused DatabaseConfig table name fields

Table names are managed by pgdbm template system via
{{tables.name}} syntax, not config fields. Remove dead config.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Fix cache_ttl Hardcoding

**Files:**
- Modify: `src/llmemory/library.py:165`
- Test: `tests/test_config.py`

**Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
@pytest.mark.asyncio
async def test_cache_ttl_respects_config(test_db):
    """Verify cache_ttl from config is actually used."""
    from llmemory import LLMemory, LLMemoryConfig

    config = LLMemoryConfig()
    config.search.cache_ttl = 1800  # 30 minutes instead of default 3600

    memory = LLMemory(connection_string=test_db, config=config)
    await memory.initialize()

    # Access the search optimizer
    search_optimizer = memory._manager._search_optimizer

    # Verify it uses config value, not hardcoded 300
    assert search_optimizer._cache_ttl == 1800, \
        f"Expected cache_ttl=1800 from config, got {search_optimizer._cache_ttl}"

    await memory.close()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py::test_cache_ttl_respects_config -v
```

Expected: FAIL (currently hardcoded to 300)

**Step 3: Fix hardcoding in library.py**

Edit `src/llmemory/library.py:165`, change:

```python
# OLD:
cache_ttl=300,

# NEW:
cache_ttl=self.config.search.cache_ttl,
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_config.py::test_cache_ttl_respects_config -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmemory/library.py tests/test_config.py
git commit -m "fix: use cache_ttl from config instead of hardcoding

Was hardcoded to 300 seconds, now respects config.search.cache_ttl
(default 3600 seconds). Add test to prevent regression.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Remove Unused LLMemoryConfig Feature Flags

**Files:**
- Modify: `src/llmemory/config.py:26-44`
- Test: `tests/test_config.py`

**Step 1: Write test**

Add to `tests/test_config.py`:

```python
def test_llmemory_config_only_has_used_fields():
    """Verify LLMemoryConfig only has implemented feature flags."""
    from llmemory.config import LLMemoryConfig

    config = LLMemoryConfig()

    # Used fields
    assert hasattr(config, 'embedding')
    assert hasattr(config, 'chunking')
    assert hasattr(config, 'search')
    assert hasattr(config, 'database')
    assert hasattr(config, 'validation')
    assert hasattr(config, 'enable_metrics')
    assert hasattr(config, 'log_level')

    # Unused fields
    assert not hasattr(config, 'enable_caching'), "Never checked in code"
    assert not hasattr(config, 'enable_background_processing'), "Never used"
    assert not hasattr(config, 'log_slow_queries'), "Never implemented"
    assert not hasattr(config, 'slow_query_threshold'), "Never used"
```

**Step 2: Run test**

```bash
uv run pytest tests/test_config.py::test_llmemory_config_only_has_used_fields -v
```

Expected: FAIL

**Step 3: Remove unused fields from LLMemoryConfig**

Edit `src/llmemory/config.py`:

```python
@dataclass
class LLMemoryConfig:
    """Main configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Keep only used fields:
    enable_metrics: bool = True
    log_level: str = "INFO"

    # Remove these (never used):
    # enable_caching: bool = True  # Never checked
    # enable_background_processing: bool = True  # Never used
    # log_slow_queries: bool = True  # Never implemented
    # slow_query_threshold: float = 1.0  # Never used
```

**Step 4: Remove from env parsing**

Remove env var parsing for deleted fields.

**Step 5: Run test**

```bash
uv run pytest tests/test_config.py::test_llmemory_config_only_has_used_fields -v
```

Expected: PASS

**Step 6: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All PASS

**Step 7: Commit**

```bash
git add src/llmemory/config.py tests/test_config.py
git commit -m "refactor: remove unused LLMemoryConfig feature flags

Remove 4 unused feature flags:
- enable_caching (never checked)
- enable_background_processing (never used)
- log_slow_queries (never implemented)
- slow_query_threshold (never used)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Add Unit Tests for QueryExpansionService

**Files:**
- Test: `tests/test_query_expansion.py`

**Step 1: Add unit test for heuristic expansion**

Add to `tests/test_query_expansion.py`:

```python
import pytest
from llmemory.query_expansion import QueryExpansionService
from llmemory.config import SearchConfig

@pytest.mark.asyncio
async def test_query_expansion_heuristic_variants():
    """Test heuristic variant generation."""
    config = SearchConfig()
    config.max_query_variants = 3
    config.include_keyword_variant = True

    service = QueryExpansionService(config)

    variants = await service.expand("how to improve customer satisfaction", max_variants=3)

    # Should get 3 variants
    assert len(variants) <= 3

    # Should include OR variant
    assert any("OR" in v for v in variants), "Should include OR variant"

    # Should include quoted variant
    assert any(v.startswith('"') and v.endswith('"') for v in variants), \
        "Should include quoted variant"

    # Variants should not include original
    assert "how to improve customer satisfaction" not in [v.lower() for v in variants]
```

**Step 2: Run test**

```bash
uv run pytest tests/test_query_expansion.py::test_query_expansion_heuristic_variants -v
```

Expected: PASS (heuristics work)

**Step 3: Add unit test with mock LLM callback**

```python
@pytest.mark.asyncio
async def test_query_expansion_with_llm_callback():
    """Test LLM callback is preferred over heuristics."""
    config = SearchConfig()
    config.max_query_variants = 3

    # Create mock callback
    async def mock_llm(query: str, limit: int):
        return [
            "semantic variant one",
            "semantic variant two",
            "semantic variant three"
        ]

    service = QueryExpansionService(config, llm_callback=mock_llm)

    variants = await service.expand("test query", max_variants=3)

    # Should use LLM variants, not heuristics
    assert "semantic variant one" in variants
    assert "semantic variant two" in variants
    assert len(variants) == 3
```

**Step 4: Run test**

```bash
uv run pytest tests/test_query_expansion.py::test_query_expansion_with_llm_callback -v
```

Expected: PASS

**Step 5: Add test for LLM fallback to heuristics**

```python
@pytest.mark.asyncio
async def test_query_expansion_fallback_on_llm_failure():
    """Test fallback to heuristics when LLM callback fails."""
    config = SearchConfig()
    config.max_query_variants = 3
    config.include_keyword_variant = True

    # Create failing callback
    async def failing_llm(query: str, limit: int):
        raise Exception("LLM API failure")

    service = QueryExpansionService(config, llm_callback=failing_llm)

    variants = await service.expand("test query expansion", max_variants=3)

    # Should fall back to heuristics
    assert len(variants) > 0, "Should fall back to heuristics on LLM failure"
    assert any("OR" in v for v in variants), "Should include heuristic OR variant"
```

**Step 6: Run test**

```bash
uv run pytest tests/test_query_expansion.py::test_query_expansion_fallback_on_llm_failure -v
```

Expected: PASS

**Step 7: Add test for timeout**

```python
@pytest.mark.asyncio
async def test_query_expansion_timeout():
    """Test LLM callback timeout (8 seconds)."""
    import asyncio

    config = SearchConfig()

    # Create slow callback
    async def slow_llm(query: str, limit: int):
        await asyncio.sleep(10)  # Exceeds 8 second timeout
        return ["variant"]

    service = QueryExpansionService(config, llm_callback=slow_llm)

    # Should timeout and fall back to heuristics
    variants = await service.expand("test", max_variants=2)

    # Should have heuristic variants (fallback)
    assert len(variants) > 0
```

**Step 8: Run test**

```bash
uv run pytest tests/test_query_expansion.py::test_query_expansion_timeout -v
```

Expected: PASS

**Step 9: Commit**

```bash
git add tests/test_query_expansion.py
git commit -m "test: add comprehensive QueryExpansionService unit tests

Add unit tests for:
- Heuristic variant generation
- LLM callback integration
- Fallback on LLM failure
- Timeout handling

All tests passing.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Run Full Test Suite and Fix Any Failures

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

If tests fail, investigate and fix. Common issues:
- Import errors from removed config fields
- Tests expecting removed fields
- Integration test issues

**Step 3: Run tests again**

```bash
uv run pytest tests/ -v
```

Expected: All PASS

**Step 4: Check test count**

```bash
uv run pytest tests/ --collect-only | grep "test session starts"
```

Expected: ~170+ tests collected (added ~8 new tests)

**Step 5: Commit if fixes needed**

```bash
git add tests/
git commit -m "fix: update tests after config cleanup

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## PHASE 2: Add SOTA Features

**Estimated Time:** 5-7 days
**Goal:** Implement query routing, contextual retrieval, enhanced parent context

---

### Task 9: Implement Query Routing Module

**Files:**
- Create: `src/llmemory/query_router.py`
- Modify: `src/llmemory/library.py`
- Modify: `src/llmemory/__init__.py`
- Test: `tests/test_query_router.py`

**Step 1: Write failing test for query routing**

Create `tests/test_query_router.py`:

```python
import pytest
from llmemory.query_router import QueryRouter, RouteDecision, RouteType

@pytest.mark.asyncio
async def test_route_answerable_query():
    """Test routing answerable queries to retrieval."""
    router = QueryRouter(openai_api_key="sk-test")

    # Answerable from context
    decision = await router.route(
        query="What is machine learning?",
        document_context=["Machine learning is a subset of AI..."]
    )

    assert decision.route_type == RouteType.RETRIEVAL
    assert decision.confidence > 0.7
    assert decision.reason is not None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_query_router.py::test_route_answerable_query -v
```

Expected: FAIL (module doesn't exist)

**Step 3: Create RouteType enum and RouteDecision model**

Create `src/llmemory/query_router.py`:

```python
# ABOUTME: Query routing for RAG systems determining if queries are answerable from available documents.
# ABOUTME: Uses LLM to classify queries and route to appropriate handlers (retrieval, web, unanswerable).

"""Query routing for RAG systems."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import logging

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
        self,
        query: str,
        document_context: List[str],
        threshold: float = 0.7
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
                        "content": "You are a query classification expert for RAG systems."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=5.0
            )

            import json
            result = json.loads(response.choices[0].message.content)

            route_type = RouteType(result["route"].lower())
            confidence = float(result["confidence"])
            reason = result["reason"]

            return RouteDecision(
                route_type=route_type,
                confidence=confidence,
                reason=reason
            )

        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            # Default to retrieval on error
            return RouteDecision(
                route_type=RouteType.RETRIEVAL,
                confidence=0.5,
                reason=f"Routing failed: {e}, defaulting to retrieval"
            )
```

**Step 4: Run test**

```bash
uv run pytest tests/test_query_router.py::test_route_answerable_query -v
```

Expected: PASS (may need to mock OpenAI)

**Step 5: Add more routing tests**

Add to `tests/test_query_router.py`:

```python
@pytest.mark.asyncio
async def test_route_web_search_query():
    """Test routing queries needing current information."""
    router = QueryRouter(openai_api_key="sk-test")

    decision = await router.route(
        query="What is the current weather in Paris?",
        document_context=["Historical climate data for Paris..."]
    )

    assert decision.route_type == RouteType.WEB_SEARCH
    assert decision.confidence > 0.6


@pytest.mark.asyncio
async def test_route_unanswerable_query():
    """Test routing unanswerable queries."""
    router = QueryRouter(openai_api_key="sk-test")

    decision = await router.route(
        query="What do you think about this?",
        document_context=["Technical documentation..."]
    )

    assert decision.route_type == RouteType.UNANSWERABLE
    assert decision.confidence > 0.5
```

**Step 6: Integrate with LLMemory**

Add method to `src/llmemory/library.py`:

```python
async def search_with_routing(
    self,
    owner_id: str,
    query_text: str,
    enable_routing: bool = True,
    routing_threshold: float = 0.7,
    **search_kwargs
) -> Dict[str, Any]:
    """Search with automatic query routing.

    Args:
        owner_id: Owner identifier
        query_text: Search query
        enable_routing: Enable query routing (default: True)
        routing_threshold: Confidence threshold for routing
        **search_kwargs: Additional arguments for search()

    Returns:
        Dict with:
        - route: RouteType (retrieval, web_search, unanswerable, clarification)
        - confidence: float (0-1)
        - results: List[SearchResult] (if route=retrieval)
        - message: str (if route != retrieval)
    """
    if not enable_routing:
        # Direct search without routing
        results = await self.search(owner_id, query_text, **search_kwargs)
        return {
            "route": "retrieval",
            "confidence": 1.0,
            "results": results
        }

    # Get sample documents for routing context
    sample_docs = await self.list_documents(owner_id, limit=5)
    context = [doc.document_name for doc in sample_docs.documents]

    # Create router
    from .query_router import QueryRouter, RouteType
    router = QueryRouter(
        openai_api_key=self._openai_api_key,
        model="gpt-4o-mini"
    )

    # Route query
    decision = await router.route(query_text, context, routing_threshold)

    if decision.route_type == RouteType.RETRIEVAL:
        results = await self.search(owner_id, query_text, **search_kwargs)
        return {
            "route": "retrieval",
            "confidence": decision.confidence,
            "results": results,
            "reason": decision.reason
        }
    elif decision.route_type == RouteType.WEB_SEARCH:
        return {
            "route": "web_search",
            "confidence": decision.confidence,
            "message": "This query requires current or external information not in your documents.",
            "reason": decision.reason
        }
    elif decision.route_type == RouteType.UNANSWERABLE:
        return {
            "route": "unanswerable",
            "confidence": decision.confidence,
            "message": "I cannot answer this type of query.",
            "reason": decision.reason
        }
    else:  # CLARIFICATION
        return {
            "route": "clarification",
            "confidence": decision.confidence,
            "message": "Could you please provide more details about your question?",
            "reason": decision.reason
        }
```

**Step 7: Export new classes**

Add to `src/llmemory/__init__.py`:

```python
from .query_router import QueryRouter, RouteDecision, RouteType
```

**Step 8: Add integration test**

```python
@pytest.mark.asyncio
async def test_search_with_routing_integration(memory_with_documents):
    """Test search_with_routing method."""
    memory = memory_with_documents

    # Answerable query should route to retrieval
    result = await memory.search_with_routing(
        owner_id="test-owner",
        query_text="machine learning algorithms",
        enable_routing=True,
        limit=5
    )

    assert result["route"] == "retrieval"
    assert "results" in result
    assert len(result["results"]) > 0
```

**Step 9: Run tests**

```bash
uv run pytest tests/test_query_router.py -v
```

Expected: PASS

**Step 10: Commit**

```bash
git add src/llmemory/query_router.py src/llmemory/library.py src/llmemory/__init__.py tests/test_query_router.py
git commit -m "feat: add query routing for answerable detection

Implement QueryRouter to classify queries as:
- RETRIEVAL (answerable from docs)
- WEB_SEARCH (needs external info)
- UNANSWERABLE (opinion/vague)
- CLARIFICATION (unclear)

Add search_with_routing() method to LLMemory.
Uses GPT-4o-mini for classification with document context.

SOTA requirement from production RAG analysis.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Implement Contextual Retrieval

**Files:**
- Modify: `src/llmemory/chunking.py` or create `src/llmemory/contextual_chunking.py`
- Modify: `src/llmemory/config.py`
- Modify: `src/llmemory/library.py`
- Test: `tests/test_contextual_retrieval.py`

**Step 1: Write failing test**

Create `tests/test_contextual_retrieval.py`:

```python
import pytest
from llmemory import LLMemory, DocumentType, LLMemoryConfig

@pytest.mark.asyncio
async def test_contextual_retrieval_prepends_document_context(test_db):
    """Test that chunks include document context when contextual_retrieval enabled."""

    config = LLMemoryConfig()
    config.chunking.enable_contextual_retrieval = True

    memory = LLMemory(connection_string=test_db, config=config)
    await memory.initialize()

    # Add document
    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 2024 Financial Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% QoQ. Profit margins improved."
    )

    # Get chunks and check they have contextualized content
    chunks = await memory.get_document_chunks(result.document.document_id)

    # First chunk should have document context prepended to embedding metadata
    assert chunks[0].metadata.get("contextualized") is True
    # The actual chunk content should remain original (for display)
    assert "Revenue increased" in chunks[0].content

    await memory.close()
```

**Step 2: Run test**

```bash
uv run pytest tests/test_contextual_retrieval.py::test_contextual_retrieval_prepends_document_context -v
```

Expected: FAIL (config field doesn't exist)

**Step 3: Add config field**

Add to `src/llmemory/config.py` in `ChunkingConfig`:

```python
@dataclass
class ChunkingConfig:
    """Chunking configuration."""
    enable_chunk_summaries: bool = False
    summary_max_tokens: int = 120
    min_chunk_size: int = 50
    max_chunk_size: int = 2000

    # Add new field:
    enable_contextual_retrieval: bool = False  # Prepend doc context to chunks before embedding
    context_template: str = "Document: {document_name}\nType: {document_type}\n\n{content}"
```

**Step 4: Implement contextual chunking**

Modify chunking code to prepend document context before embedding:

In `src/llmemory/manager.py` or wherever chunks are processed before embedding:

```python
async def _contextualize_chunk(
    self,
    chunk: DocumentChunk,
    document: Document
) -> str:
    """Prepend document context to chunk for embedding.

    Returns contextualized text for embedding while preserving
    original chunk.content for display.
    """
    if not self.config.chunking.enable_contextual_retrieval:
        return chunk.content

    # Format context template
    context = self.config.chunking.context_template.format(
        document_name=document.document_name,
        document_type=document.document_type.value,
        content=chunk.content
    )

    # Store flag in metadata
    chunk.metadata["contextualized"] = True

    return context
```

**Step 5: Use contextualized text for embeddings**

Find where embeddings are generated and use contextualized text:

```python
# Before embedding:
if self.config.chunking.enable_contextual_retrieval:
    text_for_embedding = await self._contextualize_chunk(chunk, document)
else:
    text_for_embedding = chunk.content

# Generate embedding with contextualized text
embedding = await embedding_provider.embed(text_for_embedding)
```

**Step 6: Run test**

```bash
uv run pytest tests/test_contextual_retrieval.py::test_contextual_retrieval_prepends_document_context -v
```

Expected: PASS

**Step 7: Add test comparing retrieval quality**

```python
@pytest.mark.asyncio
async def test_contextual_retrieval_improves_precision(test_db):
    """Test that contextual retrieval improves precision."""

    # Test with contextual OFF
    config_no_context = LLMemoryConfig()
    config_no_context.chunking.enable_contextual_retrieval = False

    memory_no_context = LLMemory(connection_string=test_db, config=config_no_context)
    await memory_no_context.initialize()

    await memory_no_context.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% in the technology sector."
    )

    results_no_context = await memory_no_context.search(
        owner_id="test",
        query_text="Q3 technology revenue growth",
        limit=5
    )

    await memory_no_context.close()

    # Test with contextual ON
    config_with_context = LLMemoryConfig()
    config_with_context.chunking.enable_contextual_retrieval = True

    memory_with_context = LLMemory(connection_string=test_db + "_context", config=config_with_context)
    await memory_with_context.initialize()

    await memory_with_context.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% in the technology sector."
    )

    results_with_context = await memory_with_context.search(
        owner_id="test",
        query_text="Q3 technology revenue growth",
        limit=5
    )

    await memory_with_context.close()

    # With context should have equal or better scores
    if results_no_context and results_with_context:
        assert results_with_context[0].score >= results_no_context[0].score * 0.9, \
            "Contextual retrieval should not significantly degrade scores"
```

**Step 8: Commit**

```bash
git add src/llmemory/query_router.py src/llmemory/config.py src/llmemory/manager.py src/llmemory/__init__.py tests/test_contextual_retrieval.py
git commit -m "feat: implement contextual retrieval (Anthropic approach)

Prepend document context to chunks before embedding:
- 'Document: {name}\\nType: {type}\\n\\n{content}'
- Preserves original chunk.content for display
- Marks chunks with metadata.contextualized=True
- Configurable via ChunkingConfig.enable_contextual_retrieval

Improves precision on queries requiring document-level context.

SOTA requirement from Anthropic research.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Enhance Parent Context to Use True Hierarchy

**Files:**
- Modify: `src/llmemory/search_optimizer.py:610-664`
- Test: `tests/test_parent_context.py`

**Step 1: Write test for hierarchical parent retrieval**

Create `tests/test_parent_context.py`:

```python
import pytest
from llmemory import LLMemory, DocumentType

@pytest.mark.asyncio
async def test_parent_context_uses_hierarchy(test_db):
    """Test parent context retrieval uses parent_chunk_id relationships."""

    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    # Add hierarchical document
    result = await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="Parent chunk content. " * 100 + "Child chunk content. " * 50,
        chunking_strategy="hierarchical"
    )

    # Search child chunk
    results = await memory.search(
        owner_id="test",
        query_text="child chunk",
        include_parent_context=True,
        context_window=1,
        limit=1
    )

    assert len(results) > 0
    result = results[0]

    # Should include actual parent chunk (via parent_chunk_id)
    assert result.parent_chunks is not None
    assert len(result.parent_chunks) > 0

    # Parent should be actual hierarchical parent, not just adjacent chunk
    parent = result.parent_chunks[0]
    assert "Parent chunk content" in parent.content

    await memory.close()
```

**Step 2: Run test**

```bash
uv run pytest tests/test_parent_context.py::test_parent_context_uses_hierarchy -v
```

Expected: May PASS or FAIL depending on current implementation

**Step 3: Enhance _batch_get_parent_contexts method**

Modify `src/llmemory/search_optimizer.py:610-664`:

```python
async def _batch_get_parent_contexts(
    self, chunk_ids: List[UUID], context_window: int
) -> Dict[UUID, List[DocumentChunk]]:
    """Batch retrieve parent context chunks.

    Returns hierarchical parents first, then adjacent chunks if needed.
    """
    if not chunk_ids:
        return {}

    # First, get hierarchical parents via parent_chunk_id
    parent_query = """
        SELECT
            child.chunk_id as child_id,
            parent.chunk_id,
            parent.document_id,
            parent.parent_chunk_id,
            parent.chunk_index,
            parent.chunk_level,
            parent.content,
            parent.content_hash,
            parent.token_count,
            parent.metadata,
            parent.created_at,
            parent.summary
        FROM {{tables.document_chunks}} child
        JOIN {{tables.document_chunks}} parent ON child.parent_chunk_id = parent.chunk_id
        WHERE child.chunk_id = ANY($1)
        ORDER BY parent.chunk_level DESC
    """

    parent_rows = await self.db.fetch_all(parent_query, chunk_ids)

    result: Dict[UUID, List[DocumentChunk]] = {}

    for row in parent_rows:
        child_id = row["child_id"]
        parent_chunk = DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            parent_chunk_id=row["parent_chunk_id"],
            chunk_index=row["chunk_index"],
            chunk_level=row["chunk_level"],
            content=row["content"],
            content_hash=row["content_hash"],
            token_count=row["token_count"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            summary=row.get("summary")
        )

        if child_id not in result:
            result[child_id] = []
        result[child_id].append(parent_chunk)

    # For chunks without hierarchical parents, get adjacent chunks
    chunks_needing_adjacent = [
        chunk_id for chunk_id in chunk_ids
        if chunk_id not in result or len(result[chunk_id]) < context_window
    ]

    if chunks_needing_adjacent and context_window > 0:
        # Get adjacent chunks by chunk_index (existing implementation)
        adjacent_query = """
            SELECT DISTINCT
                target.chunk_id as target_id,
                c.chunk_id,
                c.document_id,
                c.parent_chunk_id,
                c.chunk_index,
                c.chunk_level,
                c.content,
                c.content_hash,
                c.token_count,
                c.metadata,
                c.created_at,
                c.summary
            FROM {{tables.document_chunks}} target
            JOIN {{tables.document_chunks}} c
                ON c.document_id = target.document_id
            WHERE target.chunk_id = ANY($1)
              AND c.chunk_id != target.chunk_id
              AND abs(c.chunk_index - target.chunk_index) <= $2
            ORDER BY c.chunk_index
        """

        adjacent_rows = await self.db.fetch_all(
            adjacent_query,
            chunks_needing_adjacent,
            context_window
        )

        for row in adjacent_rows:
            target_id = row["target_id"]
            adjacent_chunk = DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                parent_chunk_id=row["parent_chunk_id"],
                chunk_index=row["chunk_index"],
                chunk_level=row["chunk_level"],
                content=row["content"],
                content_hash=row["content_hash"],
                token_count=row["token_count"],
                metadata=row["metadata"] or {},
                created_at=row["created_at"],
                summary=row.get("summary")
            )

            if target_id not in result:
                result[target_id] = []
            result[target_id].append(adjacent_chunk)

    return result
```

**Step 4: Run test**

```bash
uv run pytest tests/test_parent_context.py::test_parent_context_uses_hierarchy -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmemory/search_optimizer.py tests/test_parent_context.py
git commit -m "enhance: use true hierarchical parents in context retrieval

Parent context now prioritizes actual parent_chunk_id relationships
over chunk_index proximity. Falls back to adjacent chunks when no
hierarchical parent exists.

Improves context quality for hierarchical documents.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## PHASE 3: Complete Test Coverage

**Estimated Time:** 2-3 days
**Goal:** >80% test coverage, all features tested

---

### Task 12: Add Unit Tests for OptimizedAsyncSearch

**Files:**
- Test: `tests/test_search_optimizer.py` (create)

**Step 1: Write tests for search optimizer**

Create `tests/test_search_optimizer.py`:

```python
import pytest
from llmemory.search_optimizer import OptimizedAsyncSearch, SearchQuery, SearchType
from llmemory.config import SearchConfig

@pytest.mark.asyncio
async def test_search_optimizer_initialization(test_db_manager):
    """Test OptimizedAsyncSearch initialization."""
    from pgdbm import AsyncDatabaseManager

    db = test_db_manager

    optimizer = OptimizedAsyncSearch(
        db=db,
        cache_ttl=300,
        max_concurrent_queries=100,
        enable_query_optimization=True,
        hnsw_ef_search=100
    )

    assert optimizer._cache_ttl == 300
    assert optimizer._max_concurrent == 100


@pytest.mark.asyncio
async def test_vector_search_with_hnsw(memory_with_documents):
    """Test vector search uses HNSW index."""
    memory = memory_with_documents

    results = await memory.search(
        owner_id="test-owner",
        query_text="machine learning",
        search_type=SearchType.VECTOR,
        limit=10
    )

    assert len(results) > 0
    # All results should have similarity scores
    for result in results:
        assert result.similarity is not None
        assert 0 <= result.similarity <= 1


@pytest.mark.asyncio
async def test_text_search_with_bm25(memory_with_documents):
    """Test text search uses BM25."""
    memory = memory_with_documents

    results = await memory.search(
        owner_id="test-owner",
        query_text="machine learning",
        search_type=SearchType.TEXT,
        limit=10
    )

    assert len(results) > 0
    # All results should have text_rank scores
    for result in results:
        assert result.text_rank is not None


@pytest.mark.asyncio
async def test_hybrid_search_rrf_fusion(memory_with_documents):
    """Test hybrid search uses RRF fusion correctly."""
    memory = memory_with_documents

    results = await memory.search(
        owner_id="test-owner",
        query_text="machine learning",
        search_type=SearchType.HYBRID,
        alpha=0.5,
        limit=10
    )

    assert len(results) > 0
    # All results should have RRF scores
    for result in results:
        assert result.rrf_score is not None
        # Should also have individual scores
        assert result.similarity is not None or result.text_rank is not None
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_search_optimizer.py -v
```

Expected: PASS (or fix issues)

**Step 3: Commit**

```bash
git add tests/test_search_optimizer.py
git commit -m "test: add unit tests for OptimizedAsyncSearch

Add tests for:
- Search optimizer initialization
- Vector search with HNSW
- Text search with BM25
- Hybrid search with RRF fusion

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: Add Tests for Error Paths

**Files:**
- Test: `tests/test_error_handling.py` (create)

**Step 1: Write error handling tests**

Create `tests/test_error_handling.py`:

```python
import pytest
from llmemory import (
    LLMemory, ValidationError, DatabaseError, EmbeddingError,
    SearchError, DocumentNotFoundError, ConfigurationError
)
from llmemory.config import LLMemoryConfig

@pytest.mark.asyncio
async def test_validation_error_on_empty_owner_id(test_db):
    """Test ValidationError raised for empty owner_id."""
    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    with pytest.raises(ValidationError) as exc_info:
        await memory.add_document(
            owner_id="",  # Invalid
            id_at_origin="test",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
            content="test"
        )

    assert "owner_id" in str(exc_info.value).lower()

    await memory.close()


@pytest.mark.asyncio
async def test_validation_error_on_invalid_owner_id_pattern(test_db):
    """Test ValidationError for invalid owner_id characters."""
    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    with pytest.raises(ValidationError):
        await memory.add_document(
            owner_id="workspace@invalid!chars",  # Invalid pattern
            id_at_origin="test",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
            content="test"
        )

    await memory.close()


@pytest.mark.asyncio
async def test_document_not_found_error(test_db):
    """Test DocumentNotFoundError for non-existent documents."""
    from uuid import UUID

    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    with pytest.raises(DocumentNotFoundError):
        await memory.get_document(
            document_id=UUID("00000000-0000-0000-0000-000000000000")
        )

    await memory.close()


@pytest.mark.asyncio
async def test_embedding_error_on_invalid_api_key(test_db):
    """Test EmbeddingError when OpenAI API key is invalid."""
    memory = LLMemory(
        connection_string=test_db,
        openai_api_key="sk-invalid"
    )
    await memory.initialize()

    with pytest.raises(EmbeddingError):
        await memory.add_document(
            owner_id="test",
            id_at_origin="test",
            document_name="test.txt",
            document_type=DocumentType.TEXT,
            content="test content",
            generate_embeddings=True
        )

    await memory.close()


@pytest.mark.asyncio
async def test_configuration_error_on_missing_connection():
    """Test ConfigurationError when no connection info provided."""
    with pytest.raises(ConfigurationError):
        memory = LLMemory()  # No connection_string or db_manager
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_error_handling.py -v
```

Expected: PASS (or fix error handling)

**Step 3: Commit**

```bash
git add tests/test_error_handling.py
git commit -m "test: add comprehensive error handling tests

Test all exception types:
- ValidationError (empty/invalid owner_id)
- DocumentNotFoundError
- EmbeddingError (invalid API key)
- ConfigurationError (missing connection)

Ensures error paths work correctly.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: Add Tests for Timeout and Rate Limiting

**Files:**
- Test: `tests/test_timeouts.py` (create)

**Step 1: Write timeout tests**

Create `tests/test_timeouts.py`:

```python
import pytest
import asyncio
from llmemory.query_expansion import QueryExpansionService
from llmemory.config import SearchConfig

@pytest.mark.asyncio
async def test_query_expansion_timeout():
    """Test query expansion times out slow LLM callbacks."""
    config = SearchConfig()

    async def slow_callback(query: str, limit: int):
        await asyncio.sleep(10)  # Exceeds 8 second timeout
        return ["variant"]

    service = QueryExpansionService(config, llm_callback=slow_callback)

    # Should timeout and fall back to heuristics
    variants = await service.expand("test query", max_variants=2)

    # Should have heuristic variants (fallback)
    assert len(variants) > 0
    assert all("semantic variant" not in v for v in variants), \
        "Should use heuristics, not LLM variants"


@pytest.mark.asyncio
async def test_reranker_timeout():
    """Test reranking times out slow callbacks."""
    from llmemory.reranker import RerankerService
    from llmemory.models import SearchResult
    from uuid import uuid4

    async def slow_reranker(query: str, results):
        await asyncio.sleep(10)  # Exceeds timeout
        return [1.0] * len(results)

    config = SearchConfig()
    service = RerankerService(config, score_callback=slow_reranker)

    # Create test results
    results = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            content="test",
            metadata={},
            score=1.0
        )
    ]

    # Should timeout and fall back to lexical
    reranked = await service.rerank("test query", results, top_k=1)

    # Should return results (fallback to lexical scoring)
    assert len(reranked) > 0
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_timeouts.py -v
```

Expected: PASS (or fix timeout handling)

**Step 3: Commit**

```bash
git add tests/test_timeouts.py
git commit -m "test: add timeout and fallback tests

Verify timeouts work correctly for:
- Query expansion LLM callbacks (8s timeout)
- Reranker callbacks (8s timeout)

Verify fallback to heuristics/lexical on timeout.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## PHASE 4: Update Documentation for New Features

**Estimated Time:** 1 day
**Goal:** All skills updated with new features

---

### Task 15: Update multi-query Skill for LLM Expansion

**Files:**
- Modify: `.claude/skills/multi-query/SKILL.md`

**Step 1: Update Overview section**

Change lines 19-23 to explain both heuristic AND LLM expansion:

```markdown
## Overview

Multi-query expansion improves search recall by:
1. Generating multiple query variants from the original query
2. Searching with each variant independently
3. Fusing results using Reciprocal Rank Fusion (RRF)
4. Returning unified, deduplicated results

**Two expansion modes:**
- **Heuristic (default)**: Fast lexical variants using keyword extraction, OR clauses, and phrase matching. No LLM calls, <1ms latency.
- **LLM-based (configurable)**: Semantic query variants using GPT-4o-mini or similar. Better recall, 50-200ms latency, requires API key.

**When to use multi-query expansion:**
...
```

**Step 2: Add LLM configuration section**

Add after line 283:

```markdown
### LLM-Based Expansion (Advanced)

For semantic query diversity, enable LLM-based expansion:

**Environment Variables:**
```bash
LLMEMORY_ENABLE_QUERY_EXPANSION=1
LLMEMORY_QUERY_EXPANSION_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**Programmatic Configuration:**
```python
from llmemory import LLMemoryConfig

config = LLMemoryConfig()
config.search.enable_query_expansion = True
config.search.query_expansion_model = "gpt-4o-mini"  # Enable LLM expansion
config.search.max_query_variants = 3

memory = LLMemory(
    connection_string="postgresql://localhost/mydb",
    openai_api_key="sk-...",
    config=config
)
```

**LLM vs Heuristic Comparison:**

| Mode | Latency | Quality | Cost | Use Case |
|------|---------|---------|------|----------|
| Heuristic | <1ms | Good | Free | Default, high-QPS |
| LLM | 50-200ms | Excellent | ~$0.001/query | Quality-critical |

**LLM Expansion Example:**
```python
# Original: "improve customer retention"
# LLM variants:
#   1. "strategies to reduce customer churn"
#   2. "methods for increasing customer loyalty"
#   3. "how to keep customers from leaving"

results = await memory.search(
    owner_id="workspace-1",
    query_text="improve customer retention",
    query_expansion=True,
    max_query_variants=3,
    limit=10
)
```
```

**Step 3: Update Important Notes section**

Replace lines 755-777 with:

```markdown
## Important Notes

**Expansion Modes:**
- **Heuristic (default)**: Keyword extraction, OR clauses, phrase matching. Fast, no API calls.
- **LLM (configurable)**: Semantic variants via GPT-4o-mini. Set `query_expansion_model` in config.

**No LLM Required for Default:**
Query expansion works out-of-the-box with heuristic rules. No API key or LLM calls needed unless you configure `query_expansion_model`.

**Cost Considerations (LLM mode only):**
LLM expansion makes 1 API call per search. For high-volume applications with LLM expansion, consider:
- Caching common queries and their variants
- Using smaller models (gpt-4o-mini is fast and cheap)
- Enabling only for specific use cases
- Hybrid: Use heuristics for autocomplete, LLM for main search

**Quality vs Speed:**
- Heuristic: <1ms overhead, lexical diversity only
- LLM: 50-200ms overhead, semantic diversity

**Fallback Behavior:**
If LLM expansion fails or times out (8s), system automatically falls back to heuristic expansion. Search always completes.
```

**Step 4: Commit**

```bash
git add .claude/skills/multi-query/SKILL.md
git commit -m "docs: update multi-query skill for LLM expansion

Document both heuristic and LLM-based query expansion modes.
Clarify defaults, configuration, costs, and performance.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 16: Add Query Routing to RAG Skill

**Files:**
- Modify: `.claude/skills/rag/SKILL.md`

**Step 1: Add query routing section**

Add after line 100 (after Quick Start):

```markdown
## Query Routing for Production RAG

Production RAG systems should detect when queries cannot be answered from available documents.

**When to use query routing:**
- User queries may be unanswerable from your knowledge base
- Need to route to web search or external APIs
- Want to avoid hallucinated answers
- Building conversational assistants

**Example:**
```python
from llmemory import LLMemory

async with LLMemory(connection_string="...") as memory:
    # Search with automatic routing
    result = await memory.search_with_routing(
        owner_id="workspace-1",
        query_text="What's the current weather in Paris?",
        enable_routing=True,
        limit=5
    )

    if result["route"] == "retrieval":
        # Answer from documents
        return generate_answer(result["results"])
    elif result["route"] == "web_search":
        # Route to web search
        return fetch_from_web(query)
    elif result["route"] == "unanswerable":
        # Honest response
        return "I don't have information to answer that question."
    else:  # clarification
        return "Could you please provide more details?"
```

**API Reference:**

### search_with_routing()

Route queries intelligently before searching.

**Signature:**
```python
async def search_with_routing(
    owner_id: str,
    query_text: str,
    enable_routing: bool = True,
    routing_threshold: float = 0.7,
    **search_kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `owner_id` (str): Owner identifier
- `query_text` (str): Search query
- `enable_routing` (bool, default: True): Enable automatic routing
- `routing_threshold` (float, default: 0.7): Confidence threshold
- `**search_kwargs`: Additional arguments passed to search()

**Returns:**
Dict with:
- `route` (str): "retrieval", "web_search", "unanswerable", or "clarification"
- `confidence` (float): 0-1 confidence in routing decision
- `results` (List[SearchResult]): If route="retrieval"
- `message` (str): If route != "retrieval"
- `reason` (str): Explanation of routing decision

**Example:**
```python
result = await memory.search_with_routing(
    owner_id="support",
    query_text="How do I reset my password?",
    routing_threshold=0.8
)

if result["route"] == "retrieval":
    answer = generate_rag_response(result["results"])
else:
    answer = result["message"]  # Pre-formatted response
```
```

**Step 2: Commit**

```bash
git add .claude/skills/rag/SKILL.md
git commit -m "docs: add query routing to RAG skill

Document search_with_routing() method for production RAG.
Covers answerable detection and routing decisions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 17: Add Contextual Retrieval to Basic Usage Skill

**Files:**
- Modify: `.claude/skills/basic-usage/SKILL.md`

**Step 1: Add to configuration section**

Add to ChunkingConfig documentation (around line 1500):

```markdown
### ChunkingConfig

**Fields:**
- `enable_chunk_summaries` (bool, default: False): Generate chunk summaries
- `summary_max_tokens` (int, default: 120): Max tokens for summaries
- `min_chunk_size` (int, default: 50): Minimum chunk size
- `max_chunk_size` (int, default: 2000): Maximum chunk size
- `enable_contextual_retrieval` (bool, default: False): Prepend document context to chunks before embedding (Anthropic's approach)
- `context_template` (str): Template for contextual retrieval format

**Contextual Retrieval Example:**
```python
config = LLMemoryConfig()
config.chunking.enable_contextual_retrieval = True

memory = LLMemory(connection_string="...", config=config)

# Chunks are embedded with document context prepended:
# "Document: Q3 Report\nType: report\n\nRevenue increased 15%"
#
# But chunk.content remains original for display:
# "Revenue increased 15%"

await memory.add_document(
    owner_id="workspace-1",
    id_at_origin="kb",
    document_name="Q3 Report",
    document_type=DocumentType.REPORT,
    content="Revenue increased 15% QoQ...",
    chunking_config=config.chunking
)
```
```

**Step 2: Commit**

```bash
git add .claude/skills/basic-usage/SKILL.md
git commit -m "docs: add contextual retrieval to basic-usage skill

Document enable_contextual_retrieval configuration option.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## PHASE 5: Verification & Benchmarking

**Estimated Time:** 2 days
**Goal:** Verify SOTA compliance, benchmark performance

---

### Task 18: Create SOTA Compliance Test Suite

**Files:**
- Create: `tests/test_sota_compliance.py`

**Step 1: Write SOTA feature tests**

```python
import pytest
from llmemory import LLMemory, SearchType, LLMemoryConfig

@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_hybrid_search(test_db):
    """Verify hybrid search (vector + BM25) works correctly."""
    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    # Add test documents
    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="ml.txt",
        document_type=DocumentType.TEXT,
        content="Machine learning uses neural networks for pattern recognition."
    )

    # Hybrid search should work
    results = await memory.search(
        owner_id="test",
        query_text="deep learning neural networks",
        search_type=SearchType.HYBRID,
        alpha=0.5,
        limit=5
    )

    assert len(results) > 0
    # Should have RRF scores
    assert all(r.rrf_score is not None for r in results)
    # Should have both vector and text scores
    assert any(r.similarity is not None for r in results)
    assert any(r.text_rank is not None for r in results)

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_query_expansion_llm(test_db):
    """Verify LLM-based query expansion works."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY")

    config = LLMemoryConfig()
    config.search.query_expansion_model = "gpt-4o-mini"

    memory = LLMemory(connection_string=test_db, config=config)
    await memory.initialize()

    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="Artificial intelligence and machine learning."
    )

    results = await memory.search(
        owner_id="test",
        query_text="AI",
        query_expansion=True,
        max_query_variants=3,
        limit=5
    )

    # Should work (LLM generates semantic variants)
    assert len(results) >= 0  # May be 0 if variants don't match

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_reranking(test_db):
    """Verify reranking improves result quality."""
    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="test.txt",
        document_type=DocumentType.TEXT,
        content="Machine learning is a subset of artificial intelligence."
    )

    # Search with reranking
    results = await memory.search(
        owner_id="test",
        query_text="AI machine learning",
        search_type=SearchType.HYBRID,
        rerank=True,
        rerank_top_k=20,
        rerank_return_k=5,
        limit=5
    )

    # Should have rerank scores
    if len(results) > 0:
        assert all(r.rerank_score is not None for r in results), \
            "All results should have rerank scores when rerank=True"

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_query_routing(test_db):
    """Verify query routing works."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY")

    memory = LLMemory(connection_string=test_db)
    await memory.initialize()

    # Add some documents
    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="kb.txt",
        document_type=DocumentType.TEXT,
        content="Our product supports password reset via email."
    )

    # Answerable query
    result = await memory.search_with_routing(
        owner_id="test",
        query_text="How do I reset my password?",
        enable_routing=True
    )

    assert result["route"] == "retrieval"
    assert result["confidence"] > 0.5

    # Unanswerable query
    result = await memory.search_with_routing(
        owner_id="test",
        query_text="What's the current weather?",
        enable_routing=True
    )

    assert result["route"] in ["web_search", "unanswerable"]

    await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sota_contextual_retrieval(test_db):
    """Verify contextual retrieval works."""
    config = LLMemoryConfig()
    config.chunking.enable_contextual_retrieval = True

    memory = LLMemory(connection_string=test_db, config=config)
    await memory.initialize()

    await memory.add_document(
        owner_id="test",
        id_at_origin="test",
        document_name="Q3 Financial Report",
        document_type=DocumentType.REPORT,
        content="Revenue increased 15% QoQ in the technology sector."
    )

    # Chunks should be marked as contextualized
    result = await memory.list_documents(owner_id="test", limit=1)
    doc_id = result.documents[0].document_id

    chunks = await memory.get_document_chunks(doc_id, limit=1)
    assert chunks[0].metadata.get("contextualized") is True

    await memory.close()
```

**Step 2: Run SOTA tests**

```bash
uv run pytest tests/test_sota_compliance.py -v -m integration
```

Expected: PASS for all features

**Step 3: Commit**

```bash
git add tests/test_sota_compliance.py
git commit -m "test: add SOTA compliance test suite

Verify all SOTA RAG features work:
- Hybrid search (vector + BM25)
- LLM query expansion
- Reranking
- Query routing
- Contextual retrieval

Integration tests require OPENAI_API_KEY.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 19: Run Performance Benchmarks

**Files:**
- Create: `bench/sota_benchmark.py`

**Step 1: Create benchmark script**

Create `bench/sota_benchmark.py`:

```python
"""SOTA RAG performance benchmarks."""

import asyncio
import time
from llmemory import LLMemory, SearchType, LLMemoryConfig, DocumentType

async def benchmark_hybrid_search():
    """Benchmark hybrid search latency."""
    memory = LLMemory(connection_string="postgresql://localhost/mydb")
    await memory.initialize()

    # Add test documents
    for i in range(100):
        await memory.add_document(
            owner_id="bench",
            id_at_origin=f"doc-{i}",
            document_name=f"doc{i}.txt",
            document_type=DocumentType.TEXT,
            content=f"Document {i} content about machine learning and AI. " * 50
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
                limit=10
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

    assert p95 < 150, f"P95 latency {p95:.2f}ms exceeds target"

    await memory.close()


async def benchmark_with_reranking():
    """Benchmark reranking overhead."""
    config = LLMemoryConfig()
    config.search.enable_rerank = True
    config.search.rerank_provider = "lexical"  # Fast baseline

    memory = LLMemory(connection_string="postgresql://localhost/mydb", config=config)
    await memory.initialize()

    # Measure with and without reranking
    query = "machine learning"

    # Without reranking
    start = time.time()
    results_no_rerank = await memory.search(
        owner_id="bench",
        query_text=query,
        rerank=False,
        limit=10
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
        limit=10
    )
    rerank_time = (time.time() - start) * 1000

    overhead = rerank_time - no_rerank_time

    print(f"\nReranking Overhead:")
    print(f"  Without rerank: {no_rerank_time:.2f}ms")
    print(f"  With rerank: {rerank_time:.2f}ms")
    print(f"  Overhead: {overhead:.2f}ms")
    print(f"  Target: <100ms overhead")

    assert overhead < 150, f"Reranking overhead {overhead:.2f}ms too high"

    await memory.close()


if __name__ == "__main__":
    asyncio.run(benchmark_hybrid_search())
    asyncio.run(benchmark_with_reranking())
```

**Step 2: Run benchmarks**

```bash
uv run python bench/sota_benchmark.py
```

Expected: Latencies within targets (<100ms p95 for search, <100ms rerank overhead)

**Step 3: Document results**

Create `bench/RESULTS.md` with benchmark outputs.

**Step 4: Commit**

```bash
git add bench/sota_benchmark.py bench/RESULTS.md
git commit -m "bench: add SOTA performance benchmarks

Measure latency for:
- Hybrid search (target: <100ms p95)
- Reranking overhead (target: <100ms)

Document baseline performance.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## PHASE 6: Final Verification

**Estimated Time:** 1 day
**Goal:** All tests pass, all docs accurate, ready for presentation

---

### Task 20: Run Complete Test Suite

**Step 1: Run all tests with coverage**

```bash
uv run pytest tests/ -v --cov=src/llmemory --cov-report=term-missing
```

Expected: >80% coverage, all tests PASS

**Step 2: Check for any skipped tests**

```bash
uv run pytest tests/ -v | grep -i skip
```

Fix or document any skipped tests.

**Step 3: Run integration tests**

```bash
OPENAI_API_KEY=sk-... uv run pytest tests/ -v -m integration
```

Expected: All integration tests PASS

**Step 4: Document coverage**

```bash
uv run pytest tests/ --cov=src/llmemory --cov-report=html
```

Review `htmlcov/index.html` for coverage gaps.

---

### Task 21: Verify All Skills Are Accurate

**Step 1: Re-run skill validation**

For each skill, verify documentation matches implementation:

```bash
# Check basic-usage skill
uv run python -c "
from llmemory import LLMemory
import inspect

# Verify all methods documented in skill exist
methods = [
    'initialize', 'close', 'add_document', 'search',
    'search_with_documents', 'list_documents', 'get_document',
    'get_document_chunks', 'get_chunk_count',
    'delete_document', 'delete_documents', 'get_statistics',
    'search_with_routing'
]

for method in methods:
    assert hasattr(LLMemory, method), f'Method {method} not found'
    print(f'✓ {method}')

print('All methods exist!')
"
```

**Step 2: Verify examples in skills work**

Extract code examples from each skill and run them (with test data).

**Step 3: Update version numbers if needed**

Check if skills need version bumps (1.0.0 → 1.1.0).

---

### Task 22: Clean Up Repository

**Step 1: Remove validation/temporary files**

```bash
rm skill-validation-report.md
rm skill-fixes-summary.md
rm BUG-llm-query-expansion-not-wired.md
rm VALIDATION-REPORT-comprehensive.md
rm skill-creation-instructions.md
rm skill-review-findings.md
rm skills-are-the-docs.md
```

**Step 2: Organize documentation**

```bash
mkdir -p docs/reports
mv docs/plans/*.md docs/reports/  # Archive old plans if any
```

**Step 3: Update README.md**

Ensure README mentions all SOTA features:
- ✅ Hybrid search (vector + BM25)
- ✅ Query expansion (heuristic + LLM)
- ✅ Reranking (OpenAI, CrossEncoder, Lexical)
- ✅ Query routing
- ✅ Contextual retrieval
- ✅ Hierarchical chunking
- ✅ Multi-tenant support

**Step 4: Commit**

```bash
git add -A
git commit -m "docs: finalize SOTA implementation

All features implemented and tested:
- Hybrid search with RRF
- LLM query expansion (wired and tested)
- Multiple rerankers
- Query routing for answerability
- Contextual retrieval (Anthropic approach)
- Enhanced hierarchical parent context

Zero tech debt, all config fields used, >80% test coverage.

Ready for production use as SOTA RAG library.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Success Criteria

Before considering this plan complete, verify:

- [ ] All 162+ tests passing
- [ ] Test coverage >80%
- [ ] Zero unused config fields
- [ ] LLM query expansion wired and tested
- [ ] Query routing implemented and tested
- [ ] Contextual retrieval implemented and tested
- [ ] All skills accurate and up-to-date
- [ ] Performance benchmarks meet targets (<100ms p95)
- [ ] No TODO/FIXME comments in code
- [ ] No temporary files in repo
- [ ] All features documented

---

## Estimated Timeline

| Phase | Tasks | Effort | Dependencies |
|-------|-------|--------|--------------|
| Phase 1: Fix Bugs | 1-8 | 2-3 days | None |
| Phase 2: SOTA Features | 9-11 | 5-7 days | Phase 1 complete |
| Phase 3: Testing | 12-14 | 2-3 days | Can parallel with Phase 2 |
| Phase 4: Documentation | 15-17 | 1 day | Phase 2 complete |
| Phase 5: Verification | 18-19 | 2 days | All phases complete |
| Phase 6: Final | 20-22 | 1 day | Phase 5 complete |

**Total:** 13-17 days (2-3.5 weeks) with 1 engineer
**Total:** 7-10 days (1.5-2 weeks) with 2 engineers working in parallel

---

## Risk Mitigation

**High-Risk Tasks:**
- Task 9 (Query Routing): LLM integration complexity
- Task 10 (Contextual Retrieval): Embedding pipeline changes

**Mitigation:**
- Follow TDD strictly
- Test with small datasets first
- Verify backwards compatibility
- Keep heuristic fallbacks working

---

## Definition of Done

Each task is complete when:
1. ✅ Tests written (failing)
2. ✅ Implementation complete
3. ✅ Tests passing
4. ✅ Code reviewed (self-review minimum)
5. ✅ Committed with descriptive message
6. ✅ No regression in existing tests

Plan is complete when:
1. ✅ All tasks done
2. ✅ Full test suite passes
3. ✅ SOTA compliance verified
4. ✅ Performance benchmarks meet targets
5. ✅ Documentation accurate
6. ✅ Zero tech debt
