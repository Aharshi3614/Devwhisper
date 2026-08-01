"""Tests for the POST /reset endpoint.

Issue #12: Add a /reset endpoint to clear conversation memory.

These tests verify that:
  * POST /reset returns HTTP 200 with {"status": "memory cleared"}
  * The endpoint clears all conversation history
  * Subsequent requests start with no prior context
  * Existing /health route remains unaffected
"""

import pytest
from fastapi.testclient import TestClient

from main import app, conversation_sessions


@pytest.fixture(scope="module")
def client():
    """A module-scoped TestClient bound to the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Ensure conversation_sessions is empty before and after each test."""
    conversation_sessions.clear()
    yield
    conversation_sessions.clear()


def test_reset_returns_200(client):
    """POST /reset should respond with HTTP 200."""
    response = client.post("/reset")
    assert response.status_code == 200


def test_reset_returns_expected_json_body(client):
    """POST /reset should return the documented JSON payload."""
    response = client.post("/reset")
    assert response.json() == {"status": "memory cleared"}


def test_reset_status_field(client):
    """The 'status' field should be exactly 'memory cleared'."""
    response = client.post("/reset")
    assert response.json()["status"] == "memory cleared"


def test_reset_response_has_json_content_type(client):
    """The response Content-Type should be application/json."""
    response = client.post("/reset")
    assert response.headers["content-type"].startswith("application/json")


def test_reset_clears_conversation_sessions(client):
    """POST /reset should clear all entries from conversation_sessions."""
    conversation_sessions["session_a"] = {
        "history": ["User: hello\nAssistant: hi"],
        "last_used": 1000.0,
    }
    conversation_sessions["session_b"] = {
        "history": ["User: test\nAssistant: ok"],
        "last_used": 2000.0,
    }
    assert len(conversation_sessions) == 2

    client.post("/reset")

    assert len(conversation_sessions) == 0
    assert conversation_sessions == {}


def test_reset_on_empty_sessions(client):
    """POST /reset should succeed even when no sessions exist."""
    assert len(conversation_sessions) == 0

    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"status": "memory cleared"}
    assert len(conversation_sessions) == 0


def test_history_empty_after_reset(client):
    """GET /history should return no sessions after a reset."""
    conversation_sessions["session_x"] = {
        "history": ["User: q\nAssistant: a"],
        "last_used": 3000.0,
    }

    client.post("/reset")

    response = client.get("/health")
    assert response.status_code == 200

    history_resp = client.get("/history")
    assert history_resp.json() == {"session_ids": [], "sessions": []}


def test_history_lists_session_metadata(client):
    """GET /history should list sessions with preview, timestamps, and counts."""
    conversation_sessions["session_a"] = {
        "history": ["User: first question\nAssistant: first answer"],
        "last_used": 1000.0,
    }
    conversation_sessions["session_b"] = {
        "history": [
            "User: second question\nAssistant: second answer",
            "User: third question\nAssistant: third answer",
        ],
        "last_used": 2000.0,
    }

    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()

    assert data["session_ids"] == ["session_a", "session_b"]
    sessions = {s["session_id"]: s for s in data["sessions"]}
    assert sessions["session_a"] == {
        "session_id": "session_a",
        "last_used": 1000.0,
        "message_count": 1,
        "preview": "first question",
    }
    assert sessions["session_b"] == {
        "session_id": "session_b",
        "last_used": 2000.0,
        "message_count": 2,
        "preview": "second question",
    }


def test_health_unaffected_after_reset(client):
    """GET /health should continue to work after a reset."""
    conversation_sessions["temp"] = {
        "history": ["data"],
        "last_used": 100.0,
    }

    client.post("/reset")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "DevWhisper is running",
    }


def test_health_is_still_get_only(client):
    """GET /health should still reject POST with 405."""
    response = client.post("/health")
    assert response.status_code == 405
