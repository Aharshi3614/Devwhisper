"""Tests for real-time indexing progress endpoints."""

import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from indexer import progress_state


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_progress():
    progress_state.update({
        "running": False, "current": 0, "total": 0,
        "percent": 0, "current_file": "", "status": "idle", "message": "",
    })
    yield
    progress_state.update({
        "running": False, "current": 0, "total": 0,
        "percent": 0, "current_file": "", "status": "idle", "message": "",
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
    assert "message" in data


def test_progress_emits_file_count_fields(client):
    progress_state.update({
        "status": "done", "current": 3, "total": 3,
        "percent": 100, "current_file": "model.py", "message": "Done"
    })
    response = client.get("/index/progress")
    lines = [l for l in response.text.splitlines() if l.startswith("data:")]
    data = json.loads(lines[0].removeprefix("data:").strip())
    assert data["current"] == 3
    assert data["total"] == 3
    assert data["current_file"] == "model.py"


def test_progress_shows_completion_status(client):
    progress_state.update({"status": "done", "percent": 100, "message": "Indexing complete."})
    response = client.get("/index/progress")
    lines = [l for l in response.text.splitlines() if l.startswith("data:")]
    data = json.loads(lines[0].removeprefix("data:").strip())
    assert data["status"] == "done"
    assert data["percent"] == 100


def test_progress_shows_error_status(client):
    progress_state.update({"status": "error", "message": "Indexing failed: connection refused"})
    response = client.get("/index/progress")
    lines = [l for l in response.text.splitlines() if l.startswith("data:")]
    data = json.loads(lines[0].removeprefix("data:").strip())
    assert data["status"] == "error"
    assert "failed" in data["message"]


# --- progress_state unit tests ---

def test_progress_state_has_required_keys():
    required = {"running", "current", "total", "percent", "current_file", "status", "message"}
    assert required.issubset(set(progress_state.keys()))


def test_progress_percent_range():
    for pct in [0, 50, 100]:
        progress_state["percent"] = pct
        assert 0 <= progress_state["percent"] <= 100
