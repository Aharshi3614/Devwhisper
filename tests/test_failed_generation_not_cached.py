"""Regression tests for caching LLM failures (issue #293).

`llm.py` returns a readable apology when the provider call raises. Both
streaming call sites in `main.py` then treated it as a successful answer,
because the only test they applied was "non-empty":

    answer = "".join(full_response)
    if answer and answer.strip():
        cache_put(query, answer)
        update_memory(session_id, query, answer)

So one expired API key, 429 or network blip pinned the apology into the LRU.
`cache.get()` served it back for every query within CACHE_SIMILARITY_THRESHOLD
Jaccard distance and promoted it on each hit, making it the last entry to be
evicted — the failure outlived the outage that caused it.

`cache.put()`'s docstring already said only successes should reach it. Nothing
enforced that; `_persist_answer()` now does, for both call sites.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import cache
import llm
import main
from llm import GenerationStatus, LLM_ERROR_MESSAGE


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts with an empty cache and no session history."""
    cache.clear()
    main.session_manager.clear()
    yield
    cache.clear()
    main.session_manager.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def retrieval():
    """Retrieval always succeeds; these tests are about what generation does."""
    with patch.object(
        main,
        "retrieve",
        lambda query, **kwargs: ("context for " + query, ["a.py"], {"a.py": 90}),
    ):
        yield


def _failing_stream(query, context, history="", status=None):
    """Stand in for a provider that raises: apology out, status marked failed."""
    if status is not None:
        status.fail(RuntimeError("provider down"))
    yield LLM_ERROR_MESSAGE


def _working_stream(query, context, history="", status=None):
    yield f"answer to {query}"


# ---------------------------------------------------------------------------
# GenerationStatus
# ---------------------------------------------------------------------------
def test_status_starts_clean():
    status = GenerationStatus()
    assert status.failed is False
    assert status.error is None
    assert bool(status) is True


def test_status_records_the_exception():
    status = GenerationStatus()
    error = RuntimeError("boom")
    status.fail(error)

    assert status.failed is True
    assert status.error is error
    assert bool(status) is False


def test_stream_marks_status_on_provider_failure():
    status = GenerationStatus()
    with patch.object(llm, "_get_client", side_effect=RuntimeError("boom")):
        tokens = list(llm.generate_response_stream("q", "ctx", status=status))

    assert tokens == [LLM_ERROR_MESSAGE]
    assert status.failed is True
    assert isinstance(status.error, RuntimeError)


def test_stream_leaves_status_clean_on_success():
    status = GenerationStatus()

    class _Chunk:
        def __init__(self, text):
            self.choices = [
                type("C", (), {"delta": type("D", (), {"content": text})()})()
            ]

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return [_Chunk("hello "), _Chunk("world")]

    with patch.object(llm, "_get_client", return_value=_Client()):
        tokens = list(llm.generate_response_stream("q", "ctx", status=status))

    assert "".join(tokens) == "hello world"
    assert status.failed is False


def test_sync_generate_marks_status_on_failure():
    status = GenerationStatus()
    with patch.object(llm, "_get_client", side_effect=RuntimeError("boom")):
        answer = llm.generate_response("q", "ctx", status=status)

    assert answer == LLM_ERROR_MESSAGE
    assert status.failed is True


def test_status_is_optional_for_existing_callers():
    """Every pre-existing call site omits `status`; that must keep working."""
    with patch.object(llm, "_get_client", side_effect=RuntimeError("boom")):
        assert llm.generate_response("q", "ctx") == LLM_ERROR_MESSAGE
        assert list(llm.generate_response_stream("q", "ctx")) == [LLM_ERROR_MESSAGE]


# ---------------------------------------------------------------------------
# _persist_answer
# ---------------------------------------------------------------------------
def test_persist_stores_a_successful_answer():
    status = GenerationStatus()
    stored = main._persist_answer("q1", "s1", "a real answer", status, "some context")

    assert stored is True
    assert cache.get("q1") == "a real answer"
    assert "a real answer" in main.get_memory("s1")


def test_persist_refuses_a_failed_generation():
    status = GenerationStatus()
    status.fail(RuntimeError("boom"))

    stored = main._persist_answer("q1", "s1", LLM_ERROR_MESSAGE, status, "some context")

    assert stored is False
    assert cache.get("q1") is None
    assert cache.size() == 0


def test_persist_keeps_the_apology_out_of_conversation_history():
    """The apology must not be replayed into the next turn's prompt."""
    status = GenerationStatus()
    status.fail(RuntimeError("boom"))
    main._persist_answer("q1", "s1", LLM_ERROR_MESSAGE, status, "some context")

    assert main.get_memory("s1") == ""


def test_persist_refuses_an_empty_answer():
    assert (
        main._persist_answer("q1", "s1", "   ", GenerationStatus(), "ctx") is False
    )
    assert cache.size() == 0


def test_persist_does_not_cache_an_answer_built_from_no_context():
    """A "could not find this" verdict reached from nothing must not be pinned."""
    stored = main._persist_answer(
        "q1", "s1", "I could not find this in your codebase.", GenerationStatus(), ""
    )

    assert stored is False
    assert cache.get("q1") is None
    # The exchange is still remembered so the conversation reads correctly.
    assert "could not find" in main.get_memory("s1")


def test_persist_logs_the_failure(caplog):
    status = GenerationStatus()
    status.fail(ValueError("bad key"))
    with caplog.at_level("WARNING"):
        main._persist_answer("q1", "s1", LLM_ERROR_MESSAGE, status, "ctx")

    assert "not cached" in caplog.text
    assert "ValueError" in caplog.text


# ---------------------------------------------------------------------------
# End to end: /webhook
# ---------------------------------------------------------------------------
def test_webhook_failure_is_not_cached(client, retrieval):
    payload = {
        "message": {
            "type": "tool-calls",
            "call": {"id": "c1"},
            "toolCalls": [
                {
                    "id": "t1",
                    "function": {
                        "name": "query_codebase",
                        "arguments": {"query": "what does retrieve do"},
                    },
                }
            ],
        }
    }

    with patch.object(main, "generate_response_stream", _failing_stream):
        response = client.post("/webhook", json=payload)

    assert response.status_code == 200
    assert LLM_ERROR_MESSAGE in response.text
    assert cache.size() == 0


def test_webhook_recovers_once_the_provider_is_healthy(client, retrieval):
    """The poisoned entry used to survive the outage. It must not."""
    payload = {
        "message": {
            "type": "tool-calls",
            "call": {"id": "c1"},
            "toolCalls": [
                {
                    "id": "t1",
                    "function": {
                        "name": "query_codebase",
                        "arguments": {"query": "what does retrieve do"},
                    },
                }
            ],
        }
    }

    with patch.object(main, "generate_response_stream", _failing_stream):
        client.post("/webhook", json=payload)

    with patch.object(main, "generate_response_stream", _working_stream):
        response = client.post("/webhook", json=payload)

    assert "answer to what does retrieve do" in response.text
    assert LLM_ERROR_MESSAGE not in response.text


def test_failed_answer_carries_no_sources_footer(client, retrieval):
    payload = {
        "message": {
            "type": "tool-calls",
            "call": {"id": "c1"},
            "toolCalls": [
                {
                    "id": "t1",
                    "function": {
                        "name": "query_codebase",
                        "arguments": {"query": "where is the parser"},
                    },
                }
            ],
        }
    }

    with patch.object(main, "generate_response_stream", _failing_stream):
        response = client.post("/webhook", json=payload)

    assert "a.py" not in response.text


# ---------------------------------------------------------------------------
# End to end: /stream
# ---------------------------------------------------------------------------
def test_stream_failure_is_not_cached(client, retrieval):
    with patch.object(main, "generate_response_stream", _failing_stream):
        response = client.post(
            "/stream", json={"query": "how does fusion work", "sessionId": "s1"}
        )

    assert response.status_code == 200
    assert response.text == LLM_ERROR_MESSAGE
    assert cache.size() == 0
    assert main.get_memory("s1") == ""


def test_stream_success_is_still_cached(client, retrieval):
    with patch.object(main, "generate_response_stream", _working_stream):
        response = client.post(
            "/stream", json={"query": "how does fusion work", "sessionId": "s1"}
        )

    assert "answer to how does fusion work" in response.text
    assert cache.get("how does fusion work") is not None


def test_near_duplicate_query_is_not_served_a_cached_failure(client, retrieval):
    """The 0.70 Jaccard match is what turned one poisoned key into many."""
    with patch.object(main, "generate_response_stream", _failing_stream):
        client.post("/stream", json={"query": "what does retrieve do", "sessionId": "s1"})

    # Close enough to have matched the poisoned entry under the old behaviour.
    assert cache.get("what does the retrieve function do") is None

    with patch.object(main, "generate_response_stream", _working_stream):
        response = client.post(
            "/stream",
            json={"query": "what does the retrieve function do", "sessionId": "s1"},
        )

    assert LLM_ERROR_MESSAGE not in response.text
