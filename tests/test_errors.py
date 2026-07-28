"""Tests for standardized error response format across all endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app
from errors import ErrorResponse, error_response


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --- Unit tests for error_response helper ---

def test_error_response_schema_fields():
    resp = error_response(400, "Bad input")
    assert resp.status_code == 400
    body = resp.body
    import json
    data = json.loads(body)
    assert data["status"] == "error"
    assert data["code"] == 400
    assert data["message"] == "Bad input"


def test_error_response_model_defaults():
    model = ErrorResponse(code=422, message="Unprocessable")
    assert model.status == "error"
    assert model.code == 422
    assert model.message == "Unprocessable"


def test_error_response_500():
    resp = error_response(500, "Server failure")
    import json
    data = json.loads(resp.body)
    assert data == {"status": "error", "code": 500, "message": "Server failure"}


# --- Integration tests: /webhook endpoint ---

def test_webhook_bad_json_params_returns_standard_error(client):
    payload = {
        "message": {
            "type": "function-call",
            "functionCall": {
                "name": "query_codebase",
                "arguments": "not-valid-json"
            }
        }
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == 400
    assert isinstance(body["message"], str) and body["message"]


def test_webhook_empty_query_returns_standard_error(client):
    payload = {
        "message": {
            "type": "function-call",
            "functionCall": {
                "name": "query_codebase",
                "arguments": {"query": ""}
            }
        }
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == 400
    assert isinstance(body["message"], str) and body["message"]


def test_webhook_server_error_returns_standard_error(client):
    with patch("main.retrieve", side_effect=RuntimeError("db down")):
        payload = {
            "message": {
                "type": "function-call",
                "functionCall": {
                    "name": "query_codebase",
                    "arguments": {"query": "what does preprocess do?"}
                }
            }
        }
        response = client.post("/webhook", json=payload)
    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == 500
    assert isinstance(body["message"], str) and body["message"]


# --- Integration tests: /stream endpoint ---

def test_stream_missing_query_returns_standard_error(client):
    response = client.post("/stream", json={})
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == 400
    assert isinstance(body["message"], str) and body["message"]


def test_stream_server_error_returns_standard_error(client):
    with patch("main.retrieve", side_effect=RuntimeError("retrieval failed")):
        response = client.post("/stream", json={"query": "explain pipeline"})
    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == 500
    assert isinstance(body["message"], str) and body["message"]


# --- Consistency: all error responses share the same shape ---

def test_all_error_responses_have_consistent_keys(client):
    """Every error path must return {status, code, message}."""
    required_keys = {"status", "code", "message"}

    # 400 from /stream
    r1 = client.post("/stream", json={})
    assert required_keys == set(r1.json().keys())

    # 400 from /webhook (empty query)
    r2 = client.post("/webhook", json={
        "message": {
            "type": "function-call",
            "functionCall": {"name": "query_codebase", "arguments": {"query": ""}}
        }
    })
    assert required_keys == set(r2.json().keys())
