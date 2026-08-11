"""Tests for Prompt Preview Mode (Issue #216)."""
import pytest
from prompt_builder import generate_prompt_preview
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_prompt_preview_function():
    preview = generate_prompt_preview(user_query="Where is main?", context="def main(): pass")
    assert preview["user_query"] == "Where is main?"
    assert "def main(): pass" in preview["retrieved_context"]
    assert len(preview["final_prompt_messages"]) == 2

def test_prompt_preview_endpoint_disabled_by_default():
    response = client.post("/prompt/preview", json={"query": "test"})
    assert response.status_code == 403

def test_prompt_preview_endpoint_enabled():
    response = client.post("/prompt/preview", json={"query": "test", "preview_mode": True, "context": "code"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_query"] == "test"
    assert "final_prompt_messages" in data
