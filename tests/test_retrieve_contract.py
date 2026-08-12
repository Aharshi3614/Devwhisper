"""Return-contract tests for retriever.retrieve() (issue #256).

`retrieve()` had drifted away from what its callers and its own signature
promised, in two ways:

  * with ``include_sources=True`` it returned ``(context, sources)`` while
    ``main.py`` unpacked ``(context, sources, confidences)`` — so every
    cache-miss query on /stream and /webhook raised ValueError and 500'd,
    and the confidence table built a few lines earlier was discarded;
  * it advertised ``query: str | RequestContext`` but passed the object
    straight into ``preprocess_query()``, which raised TypeError.

These tests pin the contract on every path, including the empty-query early
return that is easy to forget when the shape changes again.
"""

from unittest.mock import MagicMock, patch

import pytest

import retriever
from request_context import RequestContext


def _point(file: str, score: float | None, text: str = "def f(): pass"):
    """Build a Qdrant-like point with a payload and an optional score."""
    point = MagicMock()
    point.payload = {
        "text": text,
        "file": file,
        "start_line": 1,
        "end_line": 3,
        "symbol_name": "f",
        "symbol_type": "function",
    }
    if score is None:
        # A point with no score at all (e.g. a BM25-only match).
        del point.score
    else:
        point.score = score
    return point


@pytest.fixture
def qdrant(monkeypatch):
    """Patch out Qdrant, the embedder and BM25; yield the fake Qdrant client."""
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    monkeypatch.setattr(retriever, "client", client)

    embedder = MagicMock()
    embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setattr(retriever, "embedder", embedder)

    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id: None)
    monkeypatch.setattr(retriever, "check_embedding_version", lambda: None)
    return client


# ---------------------------------------------------------------------------
# Return arity
# ---------------------------------------------------------------------------

def test_include_sources_returns_three_values(qdrant):
    """The regression that 500'd /stream: three values, not two."""
    qdrant.query_points.return_value = MagicMock(points=[_point("a.py", 0.9)])

    result = retriever.retrieve("what does f do", include_sources=True)

    assert isinstance(result, tuple)
    assert len(result) == 3, f"expected (context, sources, confidences), got {result!r}"

    context, sources, confidences = result
    assert isinstance(context, str)
    assert isinstance(sources, list)
    assert isinstance(confidences, dict)


def test_without_include_sources_returns_plain_string(qdrant):
    qdrant.query_points.return_value = MagicMock(points=[_point("a.py", 0.9)])

    result = retriever.retrieve("what does f do")

    assert isinstance(result, str)
    assert "a.py" in result


@pytest.mark.parametrize("empty_query", ["", "   ", "\n\t "])
def test_empty_query_early_return_keeps_the_same_arity(qdrant, empty_query):
    """The early return has to match — it is the path an empty query takes."""
    context, sources, confidences = retriever.retrieve(
        empty_query, include_sources=True
    )

    assert context == ""
    assert sources == []
    assert confidences == {}


def test_empty_query_without_sources_returns_empty_string(qdrant):
    assert retriever.retrieve("", include_sources=False) == ""


def test_no_results_still_returns_three_values(qdrant):
    """Qdrant returning nothing is a normal outcome, not an error path."""
    qdrant.query_points.return_value = MagicMock(points=[])

    context, sources, confidences = retriever.retrieve(
        "nothing matches this", include_sources=True
    )

    assert context == ""
    assert sources == []
    assert confidences == {}


# ---------------------------------------------------------------------------
# Confidence table
# ---------------------------------------------------------------------------

def test_confidences_cover_every_returned_source(qdrant):
    """Callers index confidences by source, so every source needs a key."""
    qdrant.query_points.return_value = MagicMock(
        points=[_point("a.py", 0.92), _point("b.py", 0.31)]
    )

    _, sources, confidences = retriever.retrieve("query", include_sources=True)

    assert set(confidences) == set(sources)
    assert confidences["a.py"] == 92
    assert confidences["b.py"] == 31


def test_missing_score_is_none_not_a_fabricated_percentage(qdrant):
    """A point with no score must not be reported as 0% confidence."""
    qdrant.query_points.return_value = MagicMock(points=[_point("a.py", None)])

    _, sources, confidences = retriever.retrieve("query", include_sources=True)

    assert sources == ["a.py"]
    assert confidences["a.py"] is None


def test_duplicate_source_keeps_the_best_score(qdrant):
    """Two chunks from one file: the higher-ranked score wins.

    `sources` is de-duplicated keeping the first occurrence, so the
    confidence table has to agree — otherwise the file is reported with the
    score of its weakest chunk.
    """
    qdrant.query_points.return_value = MagicMock(
        points=[_point("a.py", 0.95), _point("a.py", 0.20)]
    )

    _, sources, confidences = retriever.retrieve("query", include_sources=True)

    assert sources == ["a.py"], "duplicate sources should collapse to one entry"
    assert confidences["a.py"] == 95


def test_score_is_rounded_to_the_nearest_percent(qdrant):
    qdrant.query_points.return_value = MagicMock(points=[_point("a.py", 0.8449)])

    _, _, confidences = retriever.retrieve("query", include_sources=True)

    assert confidences["a.py"] == 84


# ---------------------------------------------------------------------------
# RequestContext support
# ---------------------------------------------------------------------------

def test_request_context_passed_positionally(qdrant):
    """The signature says `str | RequestContext` — so this must work."""
    qdrant.query_points.return_value = MagicMock(points=[_point("a.py", 0.5)])

    result = retriever.retrieve(RequestContext(user_query="explain f"))

    assert isinstance(result, str)
    assert "a.py" in result


def test_request_context_passed_as_keyword(qdrant):
    qdrant.query_points.return_value = MagicMock(points=[_point("a.py", 0.5)])

    result = retriever.retrieve(context=RequestContext(user_query="explain f"))

    assert "a.py" in result


def test_request_context_supplies_the_repo_id(qdrant, monkeypatch):
    """A context carrying a repo id should scope the search to it."""
    monkeypatch.setattr(
        retriever.repo_registry, "collection_name", lambda rid: f"devwhisper_{rid}"
    )
    qdrant.query_points.return_value = MagicMock(points=[])

    retriever.retrieve(RequestContext(user_query="explain f", repo_id="abc123"))

    assert qdrant.query_points.call_args.kwargs["collection_name"] == "devwhisper_abc123"


def test_explicit_repo_id_wins_over_the_context(qdrant, monkeypatch):
    """Explicit arguments beat values carried on the context."""
    monkeypatch.setattr(
        retriever.repo_registry, "collection_name", lambda rid: f"devwhisper_{rid}"
    )
    qdrant.query_points.return_value = MagicMock(points=[])

    retriever.retrieve(
        RequestContext(user_query="explain f", repo_id="from_context"),
        repo_id="explicit",
    )

    assert qdrant.query_points.call_args.kwargs["collection_name"] == "devwhisper_explicit"


def test_request_context_with_empty_query_hits_the_early_return(qdrant):
    context, sources, confidences = retriever.retrieve(
        RequestContext(user_query=""), include_sources=True
    )

    assert (context, sources, confidences) == ("", [], {})


# ---------------------------------------------------------------------------
# The callers
# ---------------------------------------------------------------------------

def test_stream_endpoint_unpacks_what_retrieve_returns():
    """End-to-end shape check against the real /stream handler.

    This is the failure users actually saw: a 500 on every uncached query.
    """
    from fastapi.testclient import TestClient

    import main

    with patch.object(main, "cache_get", return_value=None), \
            patch.object(main, "cache_put"), \
            patch.object(main, "retrieve") as mock_retrieve, \
            patch.object(main, "generate_response_stream", lambda *a, **k: iter(["answer"])):
        # Delegate to the real implementation's return shape.
        mock_retrieve.return_value = ("ctx", ["a.py"], {"a.py": 88})

        response = TestClient(main.app).post(
            "/stream", json={"query": "what does f do", "sessionId": "s1"}
        )

    assert response.status_code == 200
    body = response.text
    assert "answer" in body
    assert "a.py" in body and "88%" in body
