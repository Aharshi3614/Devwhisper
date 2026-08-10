"""Tests for Pipeline Hooks (Issue #193)."""
import pytest
from pipeline_hooks import PipelineHookRegistry, hook_registry
from retriever import retrieve
from unittest.mock import patch, MagicMock

def test_pipeline_hook_registry_execution():
    reg = PipelineHookRegistry()
    pre_called = []
    post_called = []

    def pre_fn(stage, payload):
        pre_called.append((stage, payload))

    def post_fn(stage, payload, result):
        post_called.append((stage, payload, result))

    reg.register_pre_hook("test_stage", pre_fn)
    reg.register_post_hook("test_stage", post_fn)

    reg.execute_pre_hooks("test_stage", {"key": "val"})
    reg.execute_post_hooks("test_stage", {"key": "val"}, "res")

    assert len(pre_called) == 1
    assert pre_called[0][0] == "test_stage"
    assert len(post_called) == 1
    assert post_called[0][2] == "res"

@patch("retriever.client")
@patch("retriever.embedder")
@patch("retriever._get_bm25")
def test_retrieval_stage_executes_hooks(mock_bm25, mock_embedder, mock_qdrant):
    mock_bm25.return_value = None
    mock_qdrant.query_points.return_value = MagicMock(points=[])

    events = []
    hook_registry.register_pre_hook("retrieval", lambda stage, payload: events.append("pre"))
    hook_registry.register_post_hook("retrieval", lambda stage, payload, res: events.append("post"))

    retrieve("search query")
    assert "pre" in events
    assert "post" in events
