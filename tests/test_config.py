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
