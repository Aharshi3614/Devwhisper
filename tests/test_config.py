"""Unit tests for configuration validation."""

import pytest
import config


def test_validate_config_success():
    """Valid configuration passes without raising errors."""
    config.validate_config()


def test_invalid_retrieval_top_k(monkeypatch):
    """Negative or zero RETRIEVAL_TOP_K raises a ValueError."""
    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 0)
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K must be at least 1"):
        config.validate_config()


def test_invalid_cache_similarity_threshold(monkeypatch):
    """Out-of-bound similarity threshold raises a ValueError."""
    monkeypatch.setattr(config, "CACHE_SIMILARITY_THRESHOLD", 1.5)
    with pytest.raises(ValueError, match="CACHE_SIMILARITY_THRESHOLD must be between 0.0 and 1.0"):
        config.validate_config()