"""Unit tests for the contextual query suggestions endpoint."""

import os
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
import repositories


@pytest.fixture(autouse=True)
def _default_repo(monkeypatch):
    """Pin the active repository to None so tests read the legacy root cache."""
    monkeypatch.setattr(repositories, "get_current_repo_id", lambda: None)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_suggestions_fallback_when_cache_empty(client):
    """When index cache is missing or empty, fallback to general suggestions."""
    if os.path.exists(".index_cache.json"):
        os.rename(".index_cache.json", ".index_cache.json.bak")

    try:
        response = client.get("/index/suggestions")
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 4
        # Fallbacks should contain basic questions
        for sugg in data["suggestions"]:
            assert any(term in sugg for term in ["project", "architecture", "dependencies", "set up", "run", "entry points"])
    finally:
        if os.path.exists(".index_cache.json.bak"):
            os.rename(".index_cache.json.bak", ".index_cache.json")


def test_suggestions_generated_from_cache(client):
    """When cache contains files and symbols, suggestions match codebase contents."""
    mock_cache = {
        "sample_codebase/utils.py": {
            "mtime": 123456789.0,
            "hash": "abcdef",
            "symbols": [
                {"name": "helper_func", "type": "function"},
                {"name": "ConfigHelper", "type": "class"}
            ]
        },
        "sample_codebase/README.md": {
            "mtime": 123456789.0,
            "hash": "123456",
            "symbols": []
        }
    }

    # Let's actually write the temp mock file, since it is safer and avoids complex mocks.
    if os.path.exists(".index_cache.json"):
        os.rename(".index_cache.json", ".index_cache.json.bak2")

    try:
        with open(".index_cache.json", "w", encoding="utf-8") as f:
            json.dump(mock_cache, f)

        response = client.get("/index/suggestions")
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 4

        # Suggestions should refer to our mock files and symbols
        # Note: Since selection is random, let's check that the generated suggestions contain some of our files/symbols.
        found_ref = False
        for _ in range(10):
            res = client.get("/index/suggestions").json()
            for sugg in res["suggestions"]:
                if any(x in sugg for x in ["README.md", "utils.py", "helper_func", "ConfigHelper"]):
                    found_ref = True
                    break
            if found_ref:
                break
        
        assert found_ref, "Suggestions did not contain references to codebase files or symbols from cache"

    finally:
        if os.path.exists(".index_cache.json"):
            os.remove(".index_cache.json")
        if os.path.exists(".index_cache.json.bak2"):
            os.rename(".index_cache.json.bak2", ".index_cache.json")
