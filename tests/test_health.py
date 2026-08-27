"""Tests for the /health endpoint.

Issue #49: Add a simple test for the /health endpoint.

The /health endpoint is a lightweight liveness probe used by
container orchestrators (e.g. Docker HEALTHCHECK) to decide whether
the service is up. It previously had no test coverage, so a silent
regression (e.g. someone renaming the route, changing the payload
shape, or accidentally coupling it to the Qdrant client) would not
be caught by CI.

These tests confirm that:
  * GET /health returns HTTP 200
  * The response body matches the documented contract
    {"status": "ok", "message": "DevWhisper is running"}

Per the issue, this is a test-only PR: main.py is NOT modified.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    """A module-scoped TestClient bound to the FastAPI app.

    Module scope avoids paying startup overhead per test. The
    /health endpoint has no external dependencies (no Qdrant, no
    LLM), so a plain TestClient is sufficient here.
    """
    return TestClient(app)


def test_health_returns_200(client):
    """GET /health should respond with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_expected_json_body(client):
    """GET /health should return the documented JSON payload."""
    response = client.get("/health")
    assert response.json() == {
        "status": "ok",
        "message": "DevWhisper is running",
    }


def test_health_status_field_is_ok(client):
    """The 'status' field should be exactly 'ok'."""
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_message_field_is_present(client):
    """The 'message' field should be present and a non-empty string."""
    response = client.get("/health")
    message = response.json().get("message")
    assert isinstance(message, str) and message


def test_health_is_get_only(client):
    """Only GET should be allowed; other methods should return 405.

    FastAPI auto-rejects unsupported methods with 405 Method Not
    Allowed, so this guards against accidentally broadening the
    route to POST/PUT/DELETE.
    """
    response = client.post("/health")
    assert response.status_code == 405


def test_health_response_has_json_content_type(client):
    """The response Content-Type should be application/json."""
    response = client.get("/health")
    assert response.headers["content-type"].startswith("application/json")


def test_health_diagnostics_endpoint(client, monkeypatch):
    """GET /health/diagnostics should return subsystem health breakdown."""
    import healthcheck

    monkeypatch.setattr(healthcheck, "check_qdrant", lambda: (True, "Connected"))
    monkeypatch.setattr(healthcheck, "check_embedder", lambda: (True, "384-dim"))
    monkeypatch.setattr(healthcheck, "check_llm", lambda: (True, "pong"))

    response = client.get("/health/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "subsystems" in data
    assert data["subsystems"]["qdrant"]["healthy"] is True
    assert data["subsystems"]["embedder"]["healthy"] is True
    assert data["subsystems"]["llm"]["healthy"] is True


def test_healthcheck_get_diagnostics(monkeypatch):
    """Unit test for healthcheck.get_diagnostics logic."""
    import healthcheck

    monkeypatch.setattr(healthcheck, "check_qdrant", lambda: (False, "Offline"))
    monkeypatch.setattr(healthcheck, "check_embedder", lambda: (True, "384-dim"))
    monkeypatch.setattr(healthcheck, "check_llm", lambda: (False, "Timeout"))

    diag = healthcheck.get_diagnostics()
    assert diag["status"] == "degraded"
    assert diag["subsystems"]["qdrant"]["healthy"] is False
    assert diag["subsystems"]["embedder"]["healthy"] is True
