"""tests/test_retrieval_cache.py — Unit and integration tests for multi-level retrieval caching."""

import time
import pytest
from fastapi.testclient import TestClient
from retrieval_cache import RetrievalCache, RetrievalCacheEntry
import cache


def test_retrieval_cache_entry_expiry():
    entry = RetrievalCacheEntry(
        key="test_key",
        results=[{"file_path": "main.py", "text": "def test(): pass"}],
        repo_id="repo1",
        file_paths={"main.py"},
        ttl_seconds=0.1,
    )
    assert entry.is_expired(now=entry.created_at) is False
    assert entry.is_expired(now=entry.created_at + 0.2) is True


def test_retrieval_cache_put_get_hit_miss():
    rc = RetrievalCache(max_size=10, default_ttl_seconds=60.0)
    sample_results = [
        {"file_path": "app/core.py", "text": "class Engine: pass", "score": 0.95}
    ]

    # Miss before put
    assert rc.get("how does engine work", repo_id="repo_a") is None

    # Put and hit
    rc.put("how does engine work", sample_results, repo_id="repo_a")
    hits = rc.get("how does engine work", repo_id="repo_a")
    assert hits is not None
    assert len(hits) == 1
    assert hits[0]["file_path"] == "app/core.py"

    stats = rc.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_retrieval_cache_lru_eviction():
    rc = RetrievalCache(max_size=2, default_ttl_seconds=60.0)
    res = [{"file_path": "a.py", "text": "a"}]

    rc.put("q1", res, repo_id="r1")
    rc.put("q2", res, repo_id="r1")
    assert rc.get_stats()["size"] == 2

    # Put q3 should evict q1 (oldest)
    rc.put("q3", res, repo_id="r1")
    assert rc.get_stats()["size"] == 2
    assert rc.get_stats()["evictions"] == 1
    assert rc.get("q1", repo_id="r1") is None
    assert rc.get("q2", repo_id="r1") is not None
    assert rc.get("q3", repo_id="r1") is not None


def test_retrieval_cache_file_level_invalidation():
    rc = RetrievalCache(max_size=10, default_ttl_seconds=60.0)
    res_a = [{"file_path": "src/utils.py", "text": "def util(): pass"}]
    res_b = [{"file_path": "src/models.py", "text": "class Model: pass"}]

    rc.put("query about utils", res_a, repo_id="repo_1")
    rc.put("query about models", res_b, repo_id="repo_1")

    # Invalidate utils.py
    purged = rc.invalidate_file("src/utils.py", repo_id="repo_1")
    assert purged == 1
    assert rc.get("query about utils", repo_id="repo_1") is None
    assert rc.get("query about models", repo_id="repo_1") is not None


def test_retrieval_cache_repo_invalidation():
    rc = RetrievalCache(max_size=10, default_ttl_seconds=60.0)
    res = [{"file_path": "x.py", "text": "code"}]

    rc.put("q1", res, repo_id="repo_x")
    rc.put("q2", res, repo_id="repo_y")

    purged = rc.invalidate_repo("repo_x")
    assert purged == 1
    assert rc.get("q1", repo_id="repo_x") is None
    assert rc.get("q2", repo_id="repo_y") is not None


def test_cache_stats_and_clear_endpoints(monkeypatch):
    from main import app

    client = TestClient(app)
    # Populate answer and retrieval cache
    cache.put("test question", "test answer")

    res_stats = client.get("/cache/stats")
    assert res_stats.status_code == 200
    data = res_stats.json()
    assert "answer_cache" in data
    assert "retrieval_cache" in data

    res_clear = client.post("/cache/clear")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "ok"
