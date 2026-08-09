"""Tests for the /index/summary endpoint (issue #224).

Issue #224: Add Repository Scan Summary API.

The /index/summary endpoint returns the latest indexing summary so the
frontend can display:
  * indexed file count
  * skipped file count
  * indexing duration (seconds)
  * status, percent, current_file, message, timestamp
  * detailed skipped-files list

These tests confirm the documented response contract:
  * GET /index/summary returns HTTP 200
  * Response is JSON with application/json content-type
  * All required keys are present
  * Counts default to 0 and duration/timestamp to null when no scan
    has been run yet (empty state)
  * Persisted metadata from .index_cache.json is surfaced correctly
  * Live in-progress runs override the persisted counts
  * Existing endpoints remain unaffected (no route regressions)
"""

import json
import os
import queue as queue_module
import time

import pytest
from fastapi.testclient import TestClient

from main import app, indexing_queue
from indexer import progress_state


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient bound to the FastAPI app."""
    return TestClient(app)


def _drain_queue_and_wait_for_idle(timeout: float = 2.0) -> None:
    """Drain pending jobs and wait for the background worker to go idle.

    Mirrors the helper in test_indexing_progress.py — without this, a
    job queued by a previous test can finish *after* this test sets
    progress_state, clobbering the values we are asserting on.
    """
    while True:
        try:
            indexing_queue.get_nowait()
            indexing_queue.task_done()
        except queue_module.Empty:
            break

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not progress_state.get("running"):
            return
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def reset_progress():
    """Reset progress_state around each test so they don't leak into each other."""
    _drain_queue_and_wait_for_idle()
    progress_state.update({
        "running": False, "current": 0, "total": 0,
        "percent": 0, "current_file": "", "status": "idle", "message": "",
        "skipped": [], "skipped_count": 0,
    })
    yield
    _drain_queue_and_wait_for_idle()
    progress_state.update({
        "running": False, "current": 0, "total": 0,
        "percent": 0, "current_file": "", "status": "idle", "message": "",
        "skipped": [], "skipped_count": 0,
    })


# --- Contract: HTTP basics -------------------------------------------------

def test_summary_endpoint_returns_200(client):
    response = client.get("/index/summary")
    assert response.status_code == 200


def test_summary_response_is_json(client):
    response = client.get("/index/summary")
    assert response.headers["content-type"].startswith("application/json")


def test_summary_is_get_only(client):
    """Only GET should be allowed; POST should return 405."""
    response = client.post("/index/summary")
    assert response.status_code == 405


# --- Contract: response shape ---------------------------------------------

REQUIRED_KEYS = {
    "repository_id",
    "repository_name",
    "status",
    "indexed_file_count",
    "skipped_file_count",
    "indexing_duration_seconds",
    "indexing_timestamp",
    "current_file",
    "percent",
    "message",
    "skipped_files",
}


def test_summary_has_all_required_keys(client):
    response = client.get("/index/summary")
    body = response.json()
    missing = REQUIRED_KEYS - set(body.keys())
    assert not missing, f"Missing keys: {missing}"


def test_summary_empty_state_defaults(client, monkeypatch):
    """Before any index run, counts are 0 and duration/timestamp are null."""
    # Force "no persisted metadata" regardless of whether a real
    # .index_cache.json exists in the project root (e.g. from a manual
    # index run), so the empty-state contract is deterministic.
    monkeypatch.setattr(
        "main.get_repository_metadata", lambda path: {}
    )
    import repositories
    monkeypatch.setattr(repositories, "get_current_repo_id", lambda: None)
    response = client.get("/index/summary")
    body = response.json()
    assert body["indexed_file_count"] == 0
    assert body["skipped_file_count"] == 0
    assert body["indexing_duration_seconds"] is None
    assert body["indexing_timestamp"] is None
    assert body["skipped_files"] == []
    assert body["status"] in ("idle", "done", "error", "running")


def test_summary_skipped_files_is_list(client):
    response = client.get("/index/summary")
    body = response.json()
    assert isinstance(body["skipped_files"], list)


# --- Behaviour: live progress overlay -------------------------------------

def test_summary_reflects_running_state(client):
    """When a scan is running, status should be 'running' and counts live."""
    progress_state.update({
        "running": True,
        "status": "running",
        "current": 7,
        "total": 10,
        "percent": 70,
        "current_file": "main.py",
        "skipped_count": 2,
        "skipped": [{"path": "x.py", "reason": "oversized", "detail": "too big"}],
        "message": "Indexing main.py (7/10)",
    })
    response = client.get("/index/summary")
    body = response.json()
    assert body["status"] == "running"
    assert body["indexed_file_count"] == 7
    assert body["skipped_file_count"] == 2
    assert body["percent"] == 70
    assert body["current_file"] == "main.py"
    assert body["message"] == "Indexing main.py (7/10)"
    assert len(body["skipped_files"]) == 1
    assert body["skipped_files"][0]["path"] == "x.py"


def test_summary_idle_state_when_not_running(client):
    progress_state.update({
        "running": False,
        "status": "idle",
        "current": 0,
        "total": 0,
        "percent": 0,
        "current_file": "",
        "message": "",
    })
    response = client.get("/index/summary")
    body = response.json()
    assert body["status"] == "idle"


# --- Behaviour: persisted metadata merge ----------------------------------

def test_summary_reads_persisted_cache_metadata(client, tmp_path, monkeypatch):
    """When a .index_cache.json exists, its _metadata should be surfaced."""
    cache_file = tmp_path / "index_cache_test.json"
    cache_data = {
        "src/main.py": {"mtime": 1.0, "hash": "abc"},
        "src/util.py": {"mtime": 1.0, "hash": "def"},
        "_metadata": {
            "repository_name": "test-repo",
            "indexing_timestamp": "2026-01-01T00:00:00+00:00",
            "indexing_duration_seconds": 3.14,
            "indexed_file_count": 2,
            "skipped_file_count": 1,
            "skipped_files": [
                {"path": "big.bin", "size_bytes": 9999999,
                 "reason": "oversized", "detail": "too big"},
            ],
            "embedding_model": "mock",
            "embedding_version": 1,
            "max_file_size_mb": 5,
        },
    }
    cache_file.write_text(json.dumps(cache_data))

    # Force the endpoint to read our temp cache file.
    import repositories
    monkeypatch.setattr(repositories, "get_current_repo_id", lambda: None)
    monkeypatch.setattr(
        "main.get_repository_metadata",
        lambda path: cache_data["_metadata"] if path == ".index_cache.json" else {},
    )

    # Make sure progress_state is idle so the persisted values win.
    progress_state.update({
        "running": False, "status": "idle", "current": 0, "total": 0,
        "percent": 0, "current_file": "", "message": "",
        "skipped": [], "skipped_count": 0,
    })

    response = client.get("/index/summary")
    body = response.json()
    assert body["indexed_file_count"] == 2
    assert body["skipped_file_count"] == 1
    assert body["indexing_duration_seconds"] == 3.14
    assert body["indexing_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert body["repository_name"] == "test-repo"
    assert len(body["skipped_files"]) == 1
    assert body["skipped_files"][0]["path"] == "big.bin"


# --- Regression: existing endpoints unaffected ----------------------------

def test_existing_endpoints_still_work_after_summary_added(client):
    """Adding /index/summary must not break /health, /index/queue, /statistics."""
    # /health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # /index/queue
    r = client.get("/index/queue")
    assert r.status_code == 200
    assert "jobs" in r.json()

    # /statistics — may 200 or 500 depending on Qdrant availability,
    # but the route itself must still exist (not 404).
    r = client.get("/statistics")
    assert r.status_code != 404

    # /index/progress — SSE, may 200
    progress_state.update({"status": "idle"})
    r = client.get("/index/progress")
    assert r.status_code == 200


def test_summary_route_is_registered(client):
    """The /index/summary route must be in the app's registered routes."""
    routes = {route.path for route in app.routes}
    assert "/index/summary" in routes
