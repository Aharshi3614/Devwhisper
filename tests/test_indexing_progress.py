"""Tests for real-time indexing progress endpoints."""

import json
import time
import queue as queue_module

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app, indexing_queue
from indexer import progress_state


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _drain_queue_and_wait_for_idle(timeout: float = 2.0) -> None:
    """Drain pending jobs and wait for the background worker to go idle.

    The queue worker thread (started at app startup) processes jobs
    asynchronously and toggles ``progress_state["running"]`` as each job
    starts and finishes. Without draining, a job queued by a previous
    test can finish *after* the next test sets ``running = True``,
    clobbering it back to ``False`` and causing race-dependent failures
    in tests that assert on the 409 concurrent-run rejection.

    This helper:
      1. Removes all pending jobs from the queue so the worker has
         nothing new to pick up.
      2. Polls ``progress_state["running"]`` until it becomes ``False``
         (meaning any in-progress job has finished) or the timeout
         expires.
    """
    # 1. Drain pending jobs.
    while True:
        try:
            indexing_queue.get_nowait()
            indexing_queue.task_done()
        except queue_module.Empty:
            break

    # 2. Wait for any in-progress job to finish.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not progress_state.get("running"):
            return
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def reset_progress():
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


# --- /index/start ---

def test_start_indexing_returns_200(client):
    with patch("main.threading.Thread") as mock_thread:
        mock_thread.return_value.start.return_value = None
        response = client.post("/index/start")
    assert response.status_code == 200


def test_start_indexing_returns_started_status(client):
    with patch("main.threading.Thread") as mock_thread:
        mock_thread.return_value.start.return_value = None
        response = client.post("/index/start")
    assert response.json()["status"] == "started"
    assert "message" in response.json()


def test_start_indexing_rejects_concurrent_run(client):
    progress_state["running"] = True
    response = client.post("/index/start")
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == 409
    assert "already in progress" in body["message"]


# --- /index/progress ---

def test_progress_endpoint_returns_200(client):
    progress_state.update({"status": "idle"})
    response = client.get("/index/progress")
    assert response.status_code == 200


def test_progress_response_is_event_stream(client):
    progress_state.update({"status": "idle"})
    response = client.get("/index/progress")
    assert "text/event-stream" in response.headers["content-type"]


def test_progress_emits_valid_json(client):
    progress_state.update({"status": "done", "percent": 100, "message": "Complete"})
    response = client.get("/index/progress")
    lines = [l for l in response.text.splitlines() if l.startswith("data:")]
    assert len(lines) >= 1
    data = json.loads(lines[0].removeprefix("data:").strip())
    assert "status" in data
    assert "percent" in data
