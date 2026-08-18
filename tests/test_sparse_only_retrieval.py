"""Tests for sparse-only retrieval results (issue #267).

``retrieve()`` used to fall back to ``vector_chunks[:top_k]`` whenever fewer
than two ranked lists came back non-empty. That is only correct when the one
surviving list happens to be the vector one. When dense search returned
nothing and BM25 had matches, the fallback handed back an empty list and the
query was answered from an empty context — for exactly the queries keyword
and symbol search exist to serve.

These tests cover every combination of (dense, keyword, symbol) being empty
or populated, so the selection logic cannot silently regress again.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import retriever
import vector_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _point(payload, score=0.9):
    """A Qdrant point as the client returns it."""
    return SimpleNamespace(payload=payload, score=score)


def _chunk(file, start_line=1, symbol=None, **extra):
    """A BM25/symbol chunk as it is stored in the pickled corpus."""
    doc = {
        "file": file,
        "start_line": start_line,
        "end_line": start_line + 2,
        "text": f"def {symbol or 'thing'}():\n    return 1",
        "symbol_name": symbol,
        "is_symbol": symbol is not None,
        "repository": "",
    }
    doc.update(extra)
    return doc


@pytest.fixture
def wire(monkeypatch):
    """Wire up retrieve() with controllable dense/keyword/symbol results.

    Returns a callable taking the three result lists; every one of them
    defaults to empty so each test only states what it cares about.
    """

    def _wire(points=(), keyword=(), symbol=()):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        mock_client = MagicMock()
        mock_client.query_points.return_value.points = list(points)

        monkeypatch.setattr(retriever, "embedder", mock_embedder)
        monkeypatch.setattr(retriever, "client", mock_client)
        monkeypatch.setattr(vector_store, "client", mock_client)
        # Tolerant of the argument list — retrieve() may pass the repo id through.
        monkeypatch.setattr(retriever, "check_embedding_version", lambda *a, **kw: None)

        # A non-None BM25 payload keeps the query_limit branch on its
        # hybrid path; the search functions themselves are stubbed out.
        monkeypatch.setattr(
            retriever, "_get_bm25", lambda repo_id: {"bm25": MagicMock(), "chunks": []}
        )
        monkeypatch.setattr(
            retriever,
            "_keyword_search",
            lambda *a, **kw: [dict(c, _idx=i) for i, c in enumerate(keyword)],
        )
        monkeypatch.setattr(
            retriever,
            "_exact_symbol_search",
            lambda *a, **kw: [
                dict(c, _idx=f"s_{i}", exact_match_count=1) for i, c in enumerate(symbol)
            ],
        )
        return mock_client

    return _wire


# ---------------------------------------------------------------------------
# The defect: sparse hits survive an empty dense search
# ---------------------------------------------------------------------------
def test_keyword_hits_survive_an_empty_dense_search(wire):
    """BM25 found it; dense search did not. The answer must still be there."""
    wire(points=[], keyword=[_chunk("repositories.py", 33, "make_repo_id")])

    context = retriever.retrieve("where is make_repo_id defined", top_k=5)

    assert "File: repositories.py" in context
    assert "make_repo_id" in context


def test_symbol_hits_survive_an_empty_dense_search(wire):
    """Exact symbol matching alone is enough to answer."""
    wire(points=[], symbol=[_chunk("cache.py", 244, "invalidate_repo")])

    context = retriever.retrieve("invalidate_repo", top_k=5)

    assert "File: cache.py" in context


def test_keyword_and_symbol_hits_fuse_without_any_dense_results(wire):
    """Two non-empty lists still take the RRF path when the third is empty."""
    wire(
        points=[],
        keyword=[_chunk("a.py", 1, "alpha")],
        symbol=[_chunk("b.py", 2, "beta")],
    )

    context = retriever.retrieve("alpha beta", top_k=5)

    assert "File: a.py" in context
    assert "File: b.py" in context


def test_sources_and_confidences_are_returned_for_keyword_only_hits(wire):
    """include_sources must work on the sparse-only path too."""
    wire(points=[], keyword=[_chunk("repositories.py", 33, "make_repo_id")])

    context, sources, confidences = retriever.retrieve(
        "make_repo_id", include_sources=True, top_k=5
    )

    assert sources == ["repositories.py"]
    assert set(confidences) == {"repositories.py"}
    # BM25 chunks carry no dense similarity, so the confidence is unknown
    # rather than fabricated.
    assert confidences["repositories.py"] is None
    assert context


# ---------------------------------------------------------------------------
# Everything that already worked must keep working
# ---------------------------------------------------------------------------
def test_vector_only_results_are_unchanged(wire):
    """The common case — dense search alone — behaves exactly as before."""
    wire(points=[_point({"file": "main.py", "start_line": 5, "text": "def app(): pass"})])

    context = retriever.retrieve("what does app do", top_k=5)

    assert "Result 1:" in context
    assert "File: main.py" in context


def test_all_three_retrievers_empty_returns_empty_context(wire):
    wire(points=[], keyword=[], symbol=[])

    assert retriever.retrieve("nothing matches this", top_k=5) == ""


def test_all_three_retrievers_empty_still_returns_a_three_tuple(wire):
    """The (context, sources, confidences) contract holds on the empty path."""
    wire(points=[], keyword=[], symbol=[])

    context, sources, confidences = retriever.retrieve(
        "nothing matches this", include_sources=True, top_k=5
    )

    assert context == ""
    assert sources == []
    assert confidences == {}


def test_dense_results_still_fuse_with_keyword_results(wire):
    """Two non-empty lists take the RRF path, as they always did."""
    wire(
        points=[_point({"file": "main.py", "start_line": 5, "text": "def app(): pass"})],
        keyword=[_chunk("cache.py", 10, "put")],
    )

    context = retriever.retrieve("app cache", top_k=5)

    assert "File: main.py" in context
    assert "File: cache.py" in context


# ---------------------------------------------------------------------------
# top_k is respected on every branch
# ---------------------------------------------------------------------------
def test_top_k_truncates_a_keyword_only_result_set(wire):
    wire(points=[], keyword=[_chunk(f"f{i}.py", i) for i in range(10)])

    context = retriever.retrieve("something", top_k=3)

    assert context.count("Result ") == 3
    assert "Result 4:" not in context


def test_top_k_truncates_a_symbol_only_result_set(wire):
    wire(points=[], symbol=[_chunk(f"s{i}.py", i, f"sym{i}") for i in range(8)])

    context = retriever.retrieve("sym0", top_k=2)

    assert context.count("Result ") == 2


# ---------------------------------------------------------------------------
# _fuse_ranked_lists directly
# ---------------------------------------------------------------------------
def _doc(name):
    return {"file": f"{name}.py", "start_line": 1, "text": name, "_idx": name}


def test_fuse_returns_the_single_non_empty_list_whichever_it_is():
    keyword_only = [_doc("kw")]
    symbol_only = [_doc("sym")]

    assert retriever._fuse_ranked_lists([], keyword_only, [], 5) == keyword_only
    assert retriever._fuse_ranked_lists([], [], symbol_only, 5) == symbol_only


def test_fuse_returns_empty_when_nothing_was_found():
    assert retriever._fuse_ranked_lists([], [], [], 5) == []


def test_fuse_uses_rrf_when_more_than_one_list_is_populated(monkeypatch):
    called = {}

    def fake_rrf(lists, final_top_k):
        called["lists"] = lists
        called["final_top_k"] = final_top_k
        return ["fused"]

    monkeypatch.setattr(retriever, "_rrf_fusion", fake_rrf)

    result = retriever._fuse_ranked_lists([_doc("v")], [_doc("k")], [], 7)

    assert result == ["fused"]
    assert len(called["lists"]) == 2
    assert called["final_top_k"] == 7


def test_fuse_does_not_call_rrf_for_a_single_list(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("RRF should not run for a single ranked list")

    monkeypatch.setattr(retriever, "_rrf_fusion", explode)

    assert retriever._fuse_ranked_lists([], [_doc("kw")], [], 5) == [_doc("kw")]


def test_fuse_truncates_a_single_list_to_top_k():
    docs = [_doc(str(i)) for i in range(10)]
    assert len(retriever._fuse_ranked_lists([], docs, [], 4)) == 4


def test_fuse_logs_when_only_sparse_retrievers_found_anything(monkeypatch):
    messages = []
    monkeypatch.setattr(
        retriever.logger, "debug", lambda msg, *args: messages.append(msg % args)
    )

    retriever._fuse_ranked_lists([], [_doc("kw")], [], 5)

    assert any("Dense search returned nothing" in m for m in messages)


def test_fuse_stays_quiet_when_dense_search_worked(monkeypatch):
    messages = []
    monkeypatch.setattr(
        retriever.logger, "debug", lambda msg, *args: messages.append(msg % args)
    )

    retriever._fuse_ranked_lists([_doc("v")], [], [], 5)

    assert not any("Dense search returned nothing" in m for m in messages)
