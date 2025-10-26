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
