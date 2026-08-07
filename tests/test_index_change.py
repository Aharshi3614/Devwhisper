"""Tests for the /index/change endpoint's TTL re-index detection.

The endpoint reports whether re-indexing is recommended. Re-scanning the
codebase (walking every file + hashing) is expensive, so the endpoint caches
the result and only recomputes it once every REINDEX_CHECK_INTERVAL seconds.

These tests mock `is_repository_change` so no real filesystem scan or Qdrant
connection is needed, and drive the TTL state directly through the module
globals to verify:

  * an expired TTL triggers a fresh scan;
  * a live TTL serves the cached value without scanning;
  * after expiry the cached value is refreshed;
  * startup_event records the scan time and sets the flag.
"""

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    """A bare TestClient — no lifespan, so the slow startup scan is skipped."""
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_ttl_state():
    """Reset the module globals before and after each test.

    The TTL logic lives in module globals, so tests would otherwise leak
    state into each other (e.g. a cached scan time from one test making the
    next test serve stale data).
    """
    main.reindex_recommended = False
    main._reindex_last_checked_at = 0.0
    yield
    main.reindex_recommended = False
    main._reindex_last_checked_at = 0.0


def _get_change(client, **patch_kwargs):
    """Call /index/change with is_repository_change patched.

    Returns (response, mock) so tests can assert on both the HTTP payload
    and how often the expensive scan was invoked.
    """
    with patch("main.is_repository_change", **patch_kwargs) as mock:
        response = client.get("/index/change")
        return response, mock


def test_index_change_recomputes_when_interval_elapsed(client):
    """An expired TTL triggers a fresh scan and returns its result."""
    main._reindex_last_checked_at = time.time() - main.REINDEX_CHECK_INTERVAL - 1

    response, mock = _get_change(client, return_value=True)

    assert response.status_code == 200
    mock.assert_called_once()
    assert response.json()["reindex_recommended"] is True
    assert response.json()["message"] == "Codebase changed. Re-indexing is recommended."


def test_index_change_serves_cache_before_interval(client):
    """A live TTL serves the cached value without re-scanning."""
    main.reindex_recommended = True
    main._reindex_last_checked_at = time.time()

    response, mock = _get_change(client, return_value=False)

    assert response.status_code == 200
    mock.assert_not_called()  # cache hit: the expensive scan is skipped
    assert response.json()["reindex_recommended"] is True
    assert response.json()["message"] == "Codebase changed. Re-indexing is recommended."


def test_index_change_refreshes_after_expiry(client):
    """After the TTL expires, the cached value is refreshed with a new scan."""
    main.reindex_recommended = True
    main._reindex_last_checked_at = time.time() - main.REINDEX_CHECK_INTERVAL - 5

    response, mock = _get_change(client, return_value=False)

    assert response.status_code == 200
    mock.assert_called_once()
    assert response.json()["reindex_recommended"] is False
    assert response.json()["message"] == "Index is up to date."


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_startup_records_scan_time_and_flag():
    """startup_event scans once, records the time, and sets the flag."""
    with patch("main.is_repository_change", return_value=True), \
         patch("main.embedder.encode", return_value=None):
        await main.startup_event()

    assert main.reindex_recommended is True
    assert abs(time.time() - main._reindex_last_checked_at) < 1.0
