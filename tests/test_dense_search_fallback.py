"""Regression tests for dense-search failure handling (issue #294).

`_fuse_ranked_lists()` was written to answer from BM25 and symbol matching when
dense search contributes nothing, and its docstring names the missing-collection
case by hand:

    the collection for the active repository may not exist yet while its BM25
    pickle does ... In each of those cases the sparse retrievers still hold the
    answer.

That case never worked. A missing collection is not an empty result — Qdrant's
`query_points()` raises a 404 — and the call was unguarded, so the exception
propagated out of `retrieve()` and the two sparse searches below it never ran.
The fallback added for #267 was unreachable for exactly the situation it
describes: `/stream` returned a 500 while the answer sat in the BM25 pickle.
"""

from unittest.mock import patch

import pytest

import retriever


class _Point:
    def __init__(self, payload, score=0.9):
        self.payload = payload
        self.score = score


def _chunk(text, file="a.py", line=1, symbol=None):
    return {
        "text": text,
        "file": file,
        "start_line": line,
        "end_line": line + 5,
        "symbol_name": symbol,
        "is_symbol": bool(symbol),
        "repository": "repo",
    }


@pytest.fixture(autouse=True)
def clean_caches():
    retriever.clear_bm25_cache()
    retriever.reset_metadata_cache()
    yield
    retriever.clear_bm25_cache()
    retriever.reset_metadata_cache()


@pytest.fixture
def bm25_corpus():
    """A BM25 payload whose keyword search always returns the same chunk."""
    chunks = [
        _chunk("def parse_session(): return SessionManager()", "session.py", 10),
        _chunk("def unrelated(): pass", "other.py", 3),
    ]

    class _BM25:
        def get_scores(self, tokens):
            return [4.0, 0.0]

    payload = {"bm25": _BM25(), "chunks": chunks}
    with patch.object(retriever, "_get_bm25", lambda repo_id=None: payload):
        yield payload


class _RaisingClient:
    """Stands in for Qdrant refusing the query."""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def query_points(self, **kwargs):
        self.calls += 1
        raise self.error


class _WorkingClient:
    def __init__(self, points):
        self.points = points

    def query_points(self, **kwargs):
        return type("R", (), {"points": self.points})()


# ---------------------------------------------------------------------------
# _is_missing_collection_error
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "Collection `repo_abc` doesn't exist!",
        "Collection does not exist",
        "Not found: collection repo_abc",
        "Unexpected Response: 404 (Not Found)",
    ],
)
def test_missing_collection_errors_are_recognised(message):
    assert retriever._is_missing_collection_error(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    ["timed out", "Connection refused", "Service Unavailable (503)"],
)
def test_transient_errors_are_not_mistaken_for_a_missing_collection(message):
    assert not retriever._is_missing_collection_error(RuntimeError(message))


# ---------------------------------------------------------------------------
# _dense_search
# ---------------------------------------------------------------------------
def test_dense_search_returns_ok_and_chunks_on_success():
    points = [_Point({"text": "hello", "file": "a.py", "start_line": 1}, score=0.8)]
    with patch.object(retriever, "client", _WorkingClient(points)):
        chunks, ok = retriever._dense_search("col", [0.0], None, 5)

    assert ok is True
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello"
    assert chunks[0]["score"] == 0.8
    assert chunks[0]["_idx"] == "v_0"


def test_dense_search_swallows_a_missing_collection(caplog):
    error = RuntimeError("Collection `repo_x` doesn't exist!")
    with patch.object(retriever, "client", _RaisingClient(error)):
        with caplog.at_level("WARNING"):
            chunks, ok = retriever._dense_search("repo_x", [0.0], None, 5)

    assert chunks == []
    assert ok is False
    assert "missing" in caplog.text
    assert "repo_x" in caplog.text


def test_dense_search_swallows_a_transient_failure(caplog):
    with patch.object(retriever, "client", _RaisingClient(TimeoutError("timed out"))):
        with caplog.at_level("WARNING"):
            chunks, ok = retriever._dense_search("col", [0.0], None, 5)

    assert chunks == []
    assert ok is False
    assert "falling back to sparse retrieval" in caplog.text
    assert "TimeoutError" in caplog.text


def test_dense_search_tolerates_a_point_with_no_payload():
    points = [_Point(None)]
    with patch.object(retriever, "client", _WorkingClient(points)):
        chunks, ok = retriever._dense_search("col", [0.0], None, 5)

    assert ok is True
    assert chunks[0]["file"] == "unknown"


# ---------------------------------------------------------------------------
# retrieve(): the behaviour the issue is actually about
# ---------------------------------------------------------------------------
def test_missing_collection_still_answers_from_bm25(bm25_corpus):
    """This raised before the fix; the answer was in the pickle the whole time."""
    error = RuntimeError("Collection `repo_x` doesn't exist!")
    with patch.object(retriever, "client", _RaisingClient(error)):
        context = retriever.retrieve("session manager", top_k=3)

    assert "parse_session" in context
    assert "session.py" in context


def test_unreachable_qdrant_still_answers_from_bm25(bm25_corpus):
    with patch.object(retriever, "client", _RaisingClient(ConnectionError("refused"))):
        context = retriever.retrieve("session manager", top_k=3)

    assert "parse_session" in context


def test_sources_are_returned_when_dense_search_fails(bm25_corpus):
    error = RuntimeError("Collection `repo_x` doesn't exist!")
    with patch.object(retriever, "client", _RaisingClient(error)):
        context, sources, confidences = retriever.retrieve(
            "session manager", top_k=3, include_sources=True
        )

    assert sources == ["repo:session.py"]
    # BM25 has no similarity score to report, so the confidence is None rather
    # than a fabricated number — but the key must still be present.
    assert "repo:session.py" in confidences


def test_dense_failure_does_not_break_the_three_value_contract(bm25_corpus):
    error = RuntimeError("Collection `repo_x` doesn't exist!")
    with patch.object(retriever, "client", _RaisingClient(error)):
        result = retriever.retrieve("session manager", include_sources=True)

    assert isinstance(result, tuple)
    assert len(result) == 3


def test_total_failure_is_logged_as_an_error_not_an_empty_answer(caplog):
    """Nothing matched *and* dense search failed: that is a broken index."""
    empty = {"bm25": type("B", (), {"get_scores": lambda self, t: []})(), "chunks": []}
    error = RuntimeError("Collection `repo_x` doesn't exist!")

    with patch.object(retriever, "_get_bm25", lambda repo_id=None: empty), \
            patch.object(retriever, "client", _RaisingClient(error)):
        with caplog.at_level("ERROR"):
            context = retriever.retrieve("anything", top_k=3)

    assert context == ""
    assert "the index is likely missing or unreachable" in caplog.text


def test_empty_result_from_a_healthy_index_is_not_logged_as_an_error(caplog):
    """A repository that genuinely has no answer must not look like an outage."""
    empty = {"bm25": type("B", (), {"get_scores": lambda self, t: []})(), "chunks": []}

    with patch.object(retriever, "_get_bm25", lambda repo_id=None: empty), \
            patch.object(retriever, "client", _WorkingClient([])):
        with caplog.at_level("ERROR"):
            context = retriever.retrieve("anything", top_k=3)

    assert context == ""
    assert "likely missing or unreachable" not in caplog.text


def test_dense_success_is_still_fused_with_sparse(bm25_corpus):
    """The fallback must not change the healthy path."""
    points = [
        _Point(
            {
                "text": "def parse_session(): return SessionManager()",
                "file": "session.py",
                "start_line": 10,
                "repository": "repo",
            },
            score=0.95,
        )
    ]
    with patch.object(retriever, "client", _WorkingClient(points)):
        context, sources, confidences = retriever.retrieve(
            "session manager", top_k=3, include_sources=True
        )

    assert "parse_session" in context
    # One physical chunk found by both retrievers fuses to one result.
    assert sources == ["repo:session.py"]
    assert confidences["repo:session.py"] == 95


def test_qdrant_is_only_queried_once_per_retrieve(bm25_corpus):
    error = RuntimeError("Collection `repo_x` doesn't exist!")
    failing = _RaisingClient(error)
    with patch.object(retriever, "client", failing):
        retriever.retrieve("session manager", top_k=3)

    assert failing.calls == 1
