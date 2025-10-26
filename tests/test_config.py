import pytest
from llmemory.config import SearchConfig, ChunkingConfig, LLMemoryConfig


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


def test_database_config_only_has_used_fields():
    """Verify DatabaseConfig only contains used fields."""
    from llmemory.config import DatabaseConfig

    config = DatabaseConfig()

    # Used fields
    assert hasattr(config, 'min_pool_size')
    assert hasattr(config, 'max_pool_size')
    assert hasattr(config, 'connection_timeout')
    assert hasattr(config, 'command_timeout')
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


@pytest.mark.asyncio
async def test_cache_ttl_respects_config(test_db):
    """Verify cache_ttl from config is actually used."""
    from llmemory import LLMemory, LLMemoryConfig

    config = LLMemoryConfig()
    config.search.cache_ttl = 1800  # 30 minutes instead of default 3600

    memory = LLMemory(connection_string=test_db.config.get_dsn(), config=config)
    await memory.initialize()

    # Access the optimized search
    optimized_search = memory._optimized_search

    # Verify it uses config value, not hardcoded 300
    assert optimized_search.cache_ttl == 1800, \
        f"Expected cache_ttl=1800 from config, got {optimized_search.cache_ttl}"

    await memory.close()


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
