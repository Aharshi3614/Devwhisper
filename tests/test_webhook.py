"""Tests for the /webhook Vapi handler (issue #257).

The handler had its cache/retrieval/generation block indented at function
level instead of inside the `for tool in tools:` loop that binds `query`.
Two things followed from that:

  * any message type other than assistant-request / function-call /
    tool-calls fell through to `cache_get(query)` with `query` unbound and
    came back as a 500 — Vapi sends status-update, end-of-call-report and
    friends constantly, so routine callbacks were logged as server errors;
  * with several tool calls in one payload the loop's only lasting effect
    was leaving `query` set to the last one, so every earlier call was
    silently dropped.

There were no /webhook tests at all before this, which is how it went
unnoticed. `retrieve` and the LLM are mocked throughout — this file is
about the handler's control flow, not about retrieval quality.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def pipeline():
    """Mock the retrieval + generation pipeline and record the queries seen."""
    seen = []

    def fake_retrieve(query, **kwargs):
        seen.append(query)
        return ("context for " + query, ["a.py"], {"a.py": 90})

    def fake_stream(query, context, history=""):
        yield f"answer to {query}"

    with patch.object(main, "retrieve", fake_retrieve), \
            patch.object(main, "generate_response_stream", fake_stream), \
            patch.object(main, "cache_get", return_value=None), \
            patch.object(main, "cache_put"):
        yield seen


def _tool_call(query, call_id="t1"):
    return {"id": call_id, "function": {"name": "query_codebase", "arguments": {"query": query}}}


# ---------------------------------------------------------------------------
# Unhandled message types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "msg_type",
    [
        "status-update",
        "end-of-call-report",
        "conversation-update",
        "speech-update",
        "transcript",
        "hang",
        "",
    ],
)
def test_unhandled_message_types_are_acknowledged(client, msg_type):
    """The regression: these used to 500 with an UnboundLocalError."""
    response = client.post(
        "/webhook", json={"message": {"type": msg_type, "call": {"id": "abc"}}}
    )

    assert response.status_code == 200, (
        f"{msg_type!r} should be acknowledged, not treated as a server error"
    )
    assert response.json()["status"] == "ignored"


def test_empty_body_is_acknowledged(client):
    response = client.post("/webhook", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


def test_assistant_request_still_returns_the_config(client):
    response = client.post("/webhook", json={"message": {"type": "assistant-request"}})

    assert response.status_code == 200
    assistant = response.json()["assistant"]
    assert assistant["firstMessage"]
    assert assistant["model"]["functions"][0]["name"] == "query_codebase"


# ---------------------------------------------------------------------------
# Single tool call
# ---------------------------------------------------------------------------

def test_single_tool_call_is_answered(client, pipeline):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "call": {"id": "call-1"},
                "toolCalls": [_tool_call("what does retrieve do")],
            }
        },
    )

    assert response.status_code == 200
    assert "answer to what does retrieve do" in response.text
    assert pipeline == ["what does retrieve do"]


def test_legacy_function_call_shape_is_answered(client, pipeline):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "function-call",
                "call": {"id": "call-2"},
                "functionCall": {
                    "name": "query_codebase",
                    "parameters": {"query": "explain the indexer"},
                },
            }
        },
    )

    assert response.status_code == 200
    assert "answer to explain the indexer" in response.text
    assert pipeline == ["explain the indexer"]


def test_sources_footer_is_appended(client, pipeline):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [_tool_call("where is main")],
            }
        },
    )

    assert "Sources used:" in response.text
    assert "a.py" in response.text and "90%" in response.text


def test_cache_hit_skips_retrieval(client):
    with patch.object(main, "cache_get", return_value="cached answer"), \
            patch.object(main, "retrieve") as mock_retrieve:
        response = client.post(
            "/webhook",
            json={
                "message": {
                    "type": "tool-calls",
                    "call": {"id": "call-3"},
                    "toolCalls": [_tool_call("repeat question")],
                }
            },
        )

    assert response.status_code == 200
    assert response.text == "cached answer"
    mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Multiple tool calls
# ---------------------------------------------------------------------------

def test_every_tool_call_in_the_payload_is_answered(client, pipeline):
    """The regression: only the last query used to make it through."""
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "call": {"id": "call-4"},
                "toolCalls": [
                    _tool_call("first question", "t1"),
                    _tool_call("second question", "t2"),
                    _tool_call("third question", "t3"),
                ],
            }
        },
    )

    assert response.status_code == 200
    assert pipeline == ["first question", "second question", "third question"]

    body = response.text
    for query in ("first question", "second question", "third question"):
        assert f"answer to {query}" in body


def test_multiple_answers_are_labelled_in_order(client, pipeline):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [_tool_call("q one", "t1"), _tool_call("q two", "t2")],
            }
        },
    )

    body = response.text
    assert body.index("[1/2]") < body.index("[2/2]")


def test_unknown_tool_names_are_ignored(client, pipeline):
    """An assistant may be configured with tools this server does not implement."""
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {"id": "t1", "function": {"name": "transfer_call", "arguments": {}}}
                ],
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "results": []}
    assert pipeline == []


def test_known_and_unknown_tools_mixed(client, pipeline):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {"id": "t1", "function": {"name": "end_call", "arguments": {}}},
                    _tool_call("the real question", "t2"),
                ],
            }
        },
    )

    assert response.status_code == 200
    assert pipeline == ["the real question"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_query", ["", "   "])
def test_empty_query_is_rejected(client, pipeline, bad_query):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [_tool_call(bad_query)],
            }
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["message"].lower()
    assert pipeline == []


def test_malformed_json_arguments_are_rejected(client, pipeline):
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": "t1",
                        "function": {"name": "query_codebase", "arguments": "{not json"},
                    }
                ],
            }
        },
    )

    assert response.status_code == 400
    assert "json" in response.json()["message"].lower()
    assert pipeline == []


def test_stringified_json_arguments_are_accepted(client, pipeline):
    """Vapi sends arguments as a JSON string in some configurations."""
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": "t1",
                        "function": {
                            "name": "query_codebase",
                            "arguments": '{"query": "stringified question"}',
                        },
                    }
                ],
            }
        },
    )

    assert response.status_code == 200
    assert pipeline == ["stringified question"]


def test_a_bad_call_rejects_the_whole_payload(client, pipeline):
    """Validation runs before any generation, so nothing is half-answered."""
    response = client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [_tool_call("valid question", "t1"), _tool_call("", "t2")],
            }
        },
    )

    assert response.status_code == 400
    assert pipeline == [], "no query should be run when the payload is invalid"


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

def test_answer_is_written_to_session_memory(client, pipeline):
    main.session_manager.clear()

    client.post(
        "/webhook",
        json={
            "message": {
                "type": "tool-calls",
                "call": {"id": "session-xyz"},
                "toolCalls": [_tool_call("remembered question")],
            }
        },
    )

    history = main.get_memory("session-xyz")
    assert "remembered question" in history


def test_query_extraction_helper_is_order_preserving():
    """Unit-level check of the helper, independent of the endpoint."""
    message = {
        "type": "tool-calls",
        "toolCalls": [_tool_call("a", "t1"), _tool_call("b", "t2"), _tool_call("c", "t3")],
    }

    assert main._extract_tool_queries(message) == ["a", "b", "c"]


def test_query_extraction_helper_rejects_empty_query():
    with pytest.raises(ValueError, match="empty"):
        main._extract_tool_queries(
            {"type": "tool-calls", "toolCalls": [_tool_call("")]}
        )
