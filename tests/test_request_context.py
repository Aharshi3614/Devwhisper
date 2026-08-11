"""Tests for Central Request Context Object (Issue #192)."""
import pytest
from request_context import RequestContext
from retriever import retrieve
from llm import generate_response
from unittest.mock import patch, MagicMock

def test_request_context_creation():
    ctx = RequestContext(user_query="How does indexing work?", session_id="s123", repo_id="repo1")
    assert ctx.user_query == "How does indexing work?"
    assert ctx.session_id == "s123"
    assert ctx.repo_id == "repo1"
    assert ctx.request_id is not None

def test_request_context_state_mutation():
    ctx = RequestContext(user_query="test")
    ctx.set_state("normalized_query", "test")
    assert ctx.get_state("normalized_query") == "test"
    assert ctx.get_state("non_existent", "default") == "default"

@patch("retriever.client")
@patch("retriever.embedder")
@patch("retriever._get_bm25")
def test_retriever_accepts_request_context(mock_bm25, mock_embedder, mock_qdrant):
    mock_bm25.return_value = None
    mock_qdrant.query_points.return_value = MagicMock(points=[])
    ctx = RequestContext(user_query="search query")
    res = retrieve(ctx)
    assert isinstance(res, str)

@patch("llm._get_client")
def test_llm_accepts_request_context(mock_get_client):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "mocked response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    ctx = RequestContext(user_query="explain main")
    resp = generate_response(ctx, context="def main(): pass")
    assert resp == "mocked response"
