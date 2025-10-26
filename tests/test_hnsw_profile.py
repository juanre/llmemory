import os

import pytest

from llmemory.config import HNSW_PRESETS, LLMemoryConfig, apply_hnsw_profile


def test_apply_hnsw_profile_updates_defaults():
    config = LLMemoryConfig()
    config.search.hnsw_profile = "accurate"
    apply_hnsw_profile(config)

    preset = HNSW_PRESETS["accurate"]
    assert config.database.hnsw_m == preset["m"]
    assert config.database.hnsw_ef_construction == preset["ef_construction"]
    assert config.search.hnsw_ef_search == preset["ef_search"]


def test_apply_hnsw_profile_respects_manual_overrides():
    config = LLMemoryConfig()
    config.database.hnsw_m = 64  # manual override
    config.search.hnsw_profile = "fast"
    apply_hnsw_profile(config)

    preset = HNSW_PRESETS["fast"]
    # m should remain overridden while other defaults follow the preset
    assert config.database.hnsw_m == 64
    assert config.database.hnsw_ef_construction == preset["ef_construction"]
    assert config.search.hnsw_ef_search == preset["ef_search"]


def test_env_profile(monkeypatch):
    monkeypatch.setenv("LLMEMORY_HNSW_PROFILE", "fast")
    config = LLMemoryConfig.from_env()
    apply_hnsw_profile(config)
    preset = HNSW_PRESETS["fast"]
    assert config.database.hnsw_m == preset["m"]
    assert config.database.hnsw_ef_construction == preset["ef_construction"]
    assert config.search.hnsw_ef_search == preset["ef_search"]
    monkeypatch.delenv("LLMEMORY_HNSW_PROFILE")
