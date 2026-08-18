"""Tests for Reciprocal Rank Fusion document identity (issue #266).

The three retrievers in ``retriever.py`` number their results independently:

    dense vector search  -> "v_<position in the Qdrant response>"
    BM25 keyword search  -> <integer offset into the pickled corpus>
    exact symbol search  -> "s_<the same integer offset>"

Fusing on those ids means one physical chunk lands in up to three separate
buckets, so RRF never rewards agreement between retrievers and the fused
output repeats the same code. These tests pin the behaviour to a content
identity instead: same file, same line, same symbol -> one fused result.
"""

from unittest.mock import MagicMock

import pytest

import retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chunk(file="retriever.py", start_line=10, symbol="retrieve", repo="", **extra):
    """Build a chunk dict shaped like the ones the retrievers emit."""
    doc = {
        "file": file,
        "start_line": start_line,
        "symbol_name": symbol,
        "repository": repo,
        "text": f"def {symbol}(): pass",
    }
    doc.update(extra)
    return doc


def _vector_hit(position=0, **kwargs):
    """A chunk as dense search would label it."""
    kwargs.setdefault("score", 0.9)
    return _chunk(_idx=f"v_{position}", **kwargs)


def _keyword_hit(corpus_index=0, **kwargs):
    """A chunk as BM25 search would label it — a bare integer id."""
    kwargs.setdefault("bm25_score", 7.5)
    return _chunk(_idx=corpus_index, **kwargs)


def _symbol_hit(corpus_index=0, **kwargs):
    """A chunk as exact symbol search would label it."""
    kwargs.setdefault("exact_match_count", 3)
    return _chunk(_idx=f"s_{corpus_index}", **kwargs)


def _rrf_contribution(rank, k=retriever.RRF_K):
    """The score one list at *rank* contributes to a document."""
    return 1.0 / (k + rank + 1)


# ---------------------------------------------------------------------------
# The core defect: one chunk, three ids
# ---------------------------------------------------------------------------
def test_same_chunk_from_all_three_retrievers_fuses_into_one():
    """A chunk every retriever found must appear once, not three times."""
    fused = retriever._rrf_fusion(
        [[_vector_hit()], [_keyword_hit(4)], [_symbol_hit(4)]],
        final_top_k=10,
    )

    assert len(fused) == 1, (
        f"expected one fused result, got {len(fused)}: "
        f"{[d['_idx'] for d in fused]}"
    )


def test_agreement_between_retrievers_boosts_the_score():
    """Three votes at rank 0 must outscore one vote at rank 0."""
    agreed = _vector_hit(file="agreed.py")
    fused = retriever._rrf_fusion(
        [
            [agreed],
            [_keyword_hit(4, file="agreed.py")],
            [_symbol_hit(4, file="agreed.py")],
        ],
        final_top_k=10,
    )

    assert fused[0]["rrf_score"] == pytest.approx(3 * _rrf_contribution(0))


def test_consensus_chunk_outranks_a_single_list_leader():
    """The document two retrievers agree on wins over one retriever's top hit."""
    vector_list = [
        _vector_hit(0, file="solo.py", symbol="solo"),
        _vector_hit(1, file="shared.py", symbol="shared"),
    ]
    keyword_list = [_keyword_hit(9, file="shared.py", symbol="shared")]

    fused = retriever._rrf_fusion([vector_list, keyword_list], final_top_k=5)

    assert [d["file"] for d in fused] == ["shared.py", "solo.py"]


def test_bm25_and_symbol_hits_on_the_same_corpus_entry_merge():
    """`4` and `"s_4"` address the same pickled chunk and must not split."""
    fused = retriever._rrf_fusion(
        [[_keyword_hit(4)], [_symbol_hit(4)]],
        final_top_k=10,
    )

    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(2 * _rrf_contribution(0))


# ---------------------------------------------------------------------------
# Identity boundaries — different chunks must stay separate
# ---------------------------------------------------------------------------
def test_different_files_stay_separate():
    fused = retriever._rrf_fusion(
        [[_vector_hit(0, file="a.py")], [_keyword_hit(0, file="b.py")]],
        final_top_k=10,
    )
    assert sorted(d["file"] for d in fused) == ["a.py", "b.py"]


def test_different_line_numbers_in_one_file_stay_separate():
    """Two chunks of the same file are different chunks."""
    fused = retriever._rrf_fusion(
        [
            [_vector_hit(0, start_line=10), _vector_hit(1, start_line=90)],
        ],
        final_top_k=10,
    )
    assert sorted(d["start_line"] for d in fused) == [10, 90]


def test_same_path_in_different_repositories_stays_separate():
    """`utils.py` in repoA is not `utils.py` in repoB."""
    fused = retriever._rrf_fusion(
        [
            [_vector_hit(0, file="utils.py", repo="repoA")],
            [_keyword_hit(0, file="utils.py", repo="repoB")],
        ],
        final_top_k=10,
    )
    assert len(fused) == 2
    assert sorted(d["repository"] for d in fused) == ["repoA", "repoB"]


def test_chunks_without_a_usable_file_fall_back_to_idx():
    """Location-less documents fuse on `_idx` rather than collapsing together."""
    list_a = [{"_idx": 1}, {"_idx": 2}, {"_idx": 3}]
    list_b = [{"_idx": 2}, {"_idx": 4}]

    fused = retriever._rrf_fusion([list_a, list_b], final_top_k=10)

    assert len(fused) == 4
    assert fused[0]["_idx"] == 2


def test_unknown_file_marker_is_not_treated_as_a_location():
    """`file: "unknown"` is a placeholder, not a path two chunks share."""
    fused = retriever._rrf_fusion(
        [
            [_vector_hit(0, file="unknown", symbol=None)],
            [_keyword_hit(7, file="unknown", symbol=None)],
        ],
        final_top_k=10,
    )
    assert len(fused) == 2


# ---------------------------------------------------------------------------
# Merging the fields each retriever contributes
# ---------------------------------------------------------------------------
def test_merged_document_keeps_every_retriever_score():
    """The surviving copy carries the dense score, the BM25 score and the count."""
    fused = retriever._rrf_fusion(
        [[_keyword_hit(4)], [_vector_hit(0)], [_symbol_hit(4)]],
        final_top_k=10,
    )

    doc = fused[0]
    assert doc["bm25_score"] == pytest.approx(7.5)
    assert doc["score"] == pytest.approx(0.9)
    assert doc["exact_match_count"] == 3


def test_first_list_wins_on_conflicting_values():
    """An existing value is never overwritten by a later list."""
    first = _vector_hit(0, score=0.95)
    second = _chunk(_idx=4, score=0.10)

    fused = retriever._rrf_fusion([[first], [second]], final_top_k=10)

    assert fused[0]["score"] == pytest.approx(0.95)


def test_fusion_does_not_mutate_the_input_documents():
    """Callers reuse these lists; fusion must work on copies."""
    original = _vector_hit(0)
    retriever._rrf_fusion([[original], [_keyword_hit(4)]], final_top_k=10)

    assert "rrf_score" not in original
    assert "bm25_score" not in original


# ---------------------------------------------------------------------------
# Ranking mechanics
# ---------------------------------------------------------------------------
def test_a_duplicate_inside_one_list_only_votes_once():
    """One retriever cannot compound its own vote by returning a chunk twice."""
    duplicated = [_vector_hit(0), _vector_hit(1)]  # same file/line/symbol

    fused = retriever._rrf_fusion([duplicated], final_top_k=10)

    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(_rrf_contribution(0))


def test_final_top_k_truncates_the_fused_list():
    lists = [[_vector_hit(i, file=f"f{i}.py") for i in range(10)]]
    fused = retriever._rrf_fusion(lists, final_top_k=3)
    assert len(fused) == 3


def test_ties_are_broken_by_first_appearance():
    """Equal scores must not reorder between runs."""
    # Each document sits alone at rank 0 of its own list, so all four score
    # identically and only the tie-break decides the order.
    tied = [[_vector_hit(0, file=f"f{i}.py")] for i in range(4)]

    first = [d["file"] for d in retriever._rrf_fusion(tied, final_top_k=4)]
    second = [d["file"] for d in retriever._rrf_fusion(tied, final_top_k=4)]

    assert first == second == ["f0.py", "f1.py", "f2.py", "f3.py"]


def test_empty_input_returns_empty_list():
    assert retriever._rrf_fusion([], final_top_k=5) == []
    assert retriever._rrf_fusion([[], []], final_top_k=5) == []


def test_rrf_constant_k_dampens_low_ranks():
    """A larger k flattens the gap between rank 0 and rank 1."""
    lists = [[_vector_hit(0, file="a.py"), _vector_hit(1, file="b.py")]]

    small_k = retriever._rrf_fusion(lists, k=1, final_top_k=2)
    large_k = retriever._rrf_fusion(lists, k=1000, final_top_k=2)

    small_gap = small_k[0]["rrf_score"] - small_k[1]["rrf_score"]
    large_gap = large_k[0]["rrf_score"] - large_k[1]["rrf_score"]
    assert large_gap < small_gap


# ---------------------------------------------------------------------------
# _fusion_key directly
# ---------------------------------------------------------------------------
def test_fusion_key_ignores_retriever_local_idx():
    assert retriever._fusion_key(_vector_hit(0)) == retriever._fusion_key(_keyword_hit(4))


def test_fusion_key_normalises_missing_repository_to_empty_string():
    with_none = _chunk(repo=None)
    with_empty = _chunk(repo="")
    assert retriever._fusion_key(with_none) == retriever._fusion_key(with_empty)


def test_fusion_key_is_hashable():
    """It is used as a dict key, so this is load-bearing."""
    assert isinstance(hash(retriever._fusion_key(_vector_hit(0))), int)


# ---------------------------------------------------------------------------
# End-to-end through retrieve()
# ---------------------------------------------------------------------------
def _point(payload, score=0.9):
    point = MagicMock()
    point.payload = payload
    point.score = score
    return point


def test_retrieve_does_not_repeat_a_chunk_found_by_two_retrievers(monkeypatch):
    """The formatted context must not carry the same code twice."""
    payload = {
        "file": "retriever.py",
        "start_line": 10,
        "end_line": 12,
        "text": "def retrieve(): pass",
        "symbol_name": "retrieve",
        "is_symbol": True,
        "repository": "",
    }

    response = MagicMock()
    response.points = [_point(payload)]
    monkeypatch.setattr(retriever.client, "query_points", lambda **kwargs: response)
    # Tolerant of the argument list — retrieve() may pass the repo id through.
    monkeypatch.setattr(retriever, "check_embedding_version", lambda *a, **kw: None)
    monkeypatch.setattr(
        retriever,
        "_keyword_search",
        lambda *a, **kw: [dict(payload, _idx=4, bm25_score=8.0)],
    )
    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id: {"bm25": MagicMock(), "chunks": []})
    monkeypatch.setattr(retriever, "_exact_symbol_search", lambda *a, **kw: [])

    context = retriever.retrieve("what does retrieve do", top_k=5)

    assert context.count("Result 1:") == 1
    assert "Result 2:" not in context, "the same chunk was emitted twice"
