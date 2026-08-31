"""Regression tests for the full-corpus scan in sparse retrieval (issue #296).

Two of the three retrievers touched every chunk on every query, regardless of
top_k.

`_keyword_search()` sorted all N chunks to take 20:

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

an O(N log N) sort with a Python-level key callback invoked N times, to satisfy
a HYBRID_TOP_K of 20.

`_exact_symbol_search()` enumerated the whole corpus and asked each chunk in
turn whether its symbol_name was in the requested list — a dict lookup written
as a linear scan.

Both run on the request thread before the LLM is called, and both scale with
repository size while the useful output stays fixed at 20 rows.

These tests pin the behaviour that must not change (which chunks come back, in
what order) and the work that must no longer happen.
"""


import pytest

import retriever


@pytest.fixture(autouse=True)
def clean_caches():
    retriever.clear_bm25_cache()
    yield
    retriever.clear_bm25_cache()


def _corpus(size, symbol_ratio=0.5, repository="repo"):
    chunks = []
    for i in range(size):
        is_symbol = (i % 2 == 0) if symbol_ratio else False
        chunks.append(
            {
                "text": f"def handler_{i}(value):\n    return compute(value) + {i}\n",
                "file": f"module_{i % 7}.py",
                "start_line": i * 10,
                "repository": repository,
                "is_symbol": is_symbol,
                "symbol_name": f"handler_{i}" if is_symbol else None,
            }
        )
    return chunks


class _CountingScores(list):
    """A score list that records how many times it was indexed."""

    def __init__(self, values):
        super().__init__(values)
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return super().__getitem__(index)


def _install_corpus(monkeypatch, chunks, scores=None):
    bm25 = type("B", (), {"get_scores": lambda self, tokens: scores or [0.0] * len(chunks)})()
    payload = {"bm25": bm25, "chunks": chunks}
    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id=None: payload)
    return payload


# ---------------------------------------------------------------------------
# _top_scoring_indices
# ---------------------------------------------------------------------------
def test_returns_the_highest_scores_best_first():
    scores = [0.0, 5.0, 1.0, 9.0, 0.0, 3.0]
    assert retriever._top_scoring_indices(scores, 3) == [3, 1, 5]


def test_zero_scored_chunks_are_never_returned():
    """BM25 scores 0.0 for a chunk sharing no term with the query."""
    scores = [0.0, 0.0, 2.0, 0.0]
    assert retriever._top_scoring_indices(scores, 10) == [2]


def test_all_zero_scores_return_nothing():
    assert retriever._top_scoring_indices([0.0] * 50, 20) == []


def test_negative_scores_are_excluded():
    """BM25 can return negative scores for very common terms."""
    assert retriever._top_scoring_indices([-1.0, 2.0, -0.5], 5) == [1]


def test_ties_break_by_corpus_position():
    """Matches what a stable descending sort of range(len(scores)) produced."""
    scores = [4.0, 4.0, 4.0, 1.0]
    assert retriever._top_scoring_indices(scores, 3) == [0, 1, 2]


def test_limit_of_zero_returns_nothing():
    assert retriever._top_scoring_indices([5.0, 1.0], 0) == []


def test_limit_larger_than_the_corpus_is_fine():
    assert retriever._top_scoring_indices([3.0, 1.0], 100) == [0, 1]


def test_ordering_matches_the_old_full_sort():
    """Differential check against the implementation this replaced."""
    scores = [0, 3.5, 0, 3.5, 9.1, 0, 1.2, 9.1, 0, 0.4]

    def old_way(scores, limit):
        ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ordered if scores[i] > 0][:limit]

    for limit in range(1, 11):
        assert retriever._top_scoring_indices(scores, limit) == old_way(scores, limit)


def test_the_whole_corpus_is_not_sorted():
    """The point of the change: work must be O(N), not O(N log N) with a key."""
    size = 5000
    scores = _CountingScores([0.0] * size)
    scores[10] = 7.0
    scores[4000] = 9.0
    scores.reads = 0

    retriever._top_scoring_indices(scores, 20)

    # A `sorted(range(n), key=lambda i: scores[i])` reads every element at
    # least n times over. Enumerating reads each element once, via the
    # iterator rather than __getitem__ at all.
    assert scores.reads < size


# ---------------------------------------------------------------------------
# _keyword_search
# ---------------------------------------------------------------------------
def test_keyword_search_returns_top_k_best_first(monkeypatch):
    chunks = _corpus(50)
    scores = [float(i) for i in range(50)]
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search("handler", top_k=3)

    assert [r["_idx"] for r in results] == [49, 48, 47]
    assert results[0]["bm25_score"] == 49.0


def test_keyword_search_skips_zero_scored_chunks(monkeypatch):
    chunks = _corpus(10)
    scores = [0.0] * 10
    scores[4] = 2.0
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search("handler", top_k=20)

    assert [r["_idx"] for r in results] == [4]


def test_keyword_search_still_honours_a_metadata_filter(monkeypatch):
    chunks = _corpus(40)
    scores = [float(i) for i in range(40)]
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search(
        "handler", top_k=3, metadata_filter={"file": "module_0.py"}
    )

    assert results
    assert all(r["file"] == "module_0.py" for r in results)


def test_filtering_still_fills_top_k_when_candidates_are_discarded(monkeypatch):
    """The overshoot exists so a narrow filter does not come up short."""
    chunks = _corpus(80)
    scores = [float(i) for i in range(80)]
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search(
        "handler", top_k=5, metadata_filter={"file": "module_3.py"}
    )

    assert len(results) == 5
    assert all(r["file"] == "module_3.py" for r in results)


def test_a_very_narrow_filter_still_fills_top_k(monkeypatch):
    """One matching file in a large corpus: widening has to keep looking."""
    chunks = _corpus(2000)
    scores = [float(i) for i in range(2000)]
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search(
        "handler", top_k=6, metadata_filter={"file": "module_1.py"}
    )

    assert len(results) == 6
    assert all(r["file"] == "module_1.py" for r in results)
    # Still ordered best-first.
    assert [r["bm25_score"] for r in results] == sorted(
        (r["bm25_score"] for r in results), reverse=True
    )


def test_widening_stops_when_the_positive_pool_is_exhausted(monkeypatch):
    """Fewer matches than top_k must return what exists, not loop forever."""
    chunks = _corpus(60)
    scores = [0.0] * 60
    for i in (7, 21, 35):
        scores[i] = float(i)
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search(
        "handler", top_k=20, metadata_filter={"file": "module_0.py"}
    )

    assert all(r["file"] == "module_0.py" for r in results)
    assert len(results) < 20


def test_no_filter_means_no_widening(monkeypatch):
    """The unfiltered path must stay a single pass over the score list."""
    chunks = _corpus(100)
    scores = [float(i) for i in range(100)]
    payload = _install_corpus(monkeypatch, chunks, scores)

    calls = []
    real = retriever._top_scoring_indices
    monkeypatch.setattr(
        retriever,
        "_top_scoring_indices",
        lambda s, limit: calls.append(limit) or real(s, limit),
    )

    results = retriever._keyword_search("handler", top_k=5)

    assert len(results) == 5
    assert calls == [5]
    assert payload is not None


def test_keyword_search_honours_a_repository_filter(monkeypatch):
    chunks = _corpus(10, repository="alpha") + _corpus(10, repository="beta")
    scores = [float(i) for i in range(20)]
    _install_corpus(monkeypatch, chunks, scores)

    results = retriever._keyword_search(
        "handler", top_k=5, repository_names=["alpha"]
    )

    assert results
    assert all(r["repository"] == "alpha" for r in results)


def test_keyword_search_does_not_mutate_the_cached_corpus(monkeypatch):
    """bm25_score / _idx must land on a copy, not on the cached chunk."""
    chunks = _corpus(5)
    _install_corpus(monkeypatch, chunks, [1.0] * 5)

    retriever._keyword_search("handler", top_k=5)

    assert all("bm25_score" not in chunk for chunk in chunks)
    assert all("_idx" not in chunk for chunk in chunks)


def test_keyword_search_with_no_index_returns_nothing(monkeypatch):
    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id=None: None)
    assert retriever._keyword_search("anything") == []


# ---------------------------------------------------------------------------
# The symbol index
# ---------------------------------------------------------------------------
def test_symbol_index_separates_named_symbols_from_the_rest():
    chunks = _corpus(6)
    by_name, text_only = retriever._get_symbol_index("r1", chunks)

    assert by_name["handler_0"] == [0]
    assert by_name["handler_4"] == [4]
    assert text_only == [1, 3, 5]


def test_symbol_index_groups_a_repeated_name():
    """One symbol split into parts appears as several chunks under one name."""
    chunks = [
        {"text": "a", "is_symbol": True, "symbol_name": "Widget", "symbol_part": 1},
        {"text": "b", "is_symbol": True, "symbol_name": "Widget", "symbol_part": 2},
    ]
    by_name, text_only = retriever._get_symbol_index("r1", chunks)

    assert by_name["widget"] == [0, 1]
    assert text_only == []


def test_symbol_index_is_cached_per_corpus():
    chunks = _corpus(4)
    first = retriever._get_symbol_index("r1", chunks)
    second = retriever._get_symbol_index("r1", chunks)

    assert first[0] is second[0]


def test_symbol_index_rebuilds_when_the_corpus_is_replaced():
    """_get_bm25() swaps the payload when the pickle changes on disk."""
    first = retriever._get_symbol_index("r1", _corpus(4))
    second = retriever._get_symbol_index("r1", _corpus(8))

    assert first[0] is not second[0]
    assert len(second[0]) > len(first[0])


def test_symbol_index_survives_a_freed_corpus_address_being_reused():
    """A cache keyed on id() would hand back a stale index here.

    CPython reuses the address of a freed object, so a corpus allocated after
    the previous one is dropped can land on the same id — and inherit an index
    whose indices point into a corpus that no longer exists.
    """
    retriever._get_symbol_index("r1", _corpus(4))
    for _ in range(20):
        replacement = _corpus(8)
        by_name, text_only = retriever._get_symbol_index("r1", replacement)
        assert len(by_name) + len(text_only) == len(replacement)
        assert max(text_only, default=0) < len(replacement)


def test_symbol_index_indices_always_address_the_current_corpus(monkeypatch):
    """The stale-index failure mode, end to end."""
    first = _corpus(4)
    _install_corpus(monkeypatch, first)
    assert retriever._exact_symbol_search(["handler_0"], top_k=5)

    second = [
        {"text": "x", "is_symbol": True, "symbol_name": "brand_new", "file": "z.py"}
    ]
    _install_corpus(monkeypatch, second)

    assert retriever._exact_symbol_search(["handler_0"], top_k=5) == []
    results = retriever._exact_symbol_search(["brand_new"], top_k=5)
    assert [r["file"] for r in results] == ["z.py"]


def test_clearing_the_bm25_cache_clears_the_symbol_index():
    retriever._get_symbol_index("r1", _corpus(4))
    retriever.clear_bm25_cache()

    assert retriever._symbol_index == {}


def test_invalidating_one_repository_leaves_the_others(monkeypatch):
    retriever._get_symbol_index("r1", _corpus(4))
    retriever._get_symbol_index("r2", _corpus(4))

    retriever.invalidate_bm25_cache("r1")

    assert "r1" not in retriever._symbol_index
    assert "r2" in retriever._symbol_index


# ---------------------------------------------------------------------------
# _exact_symbol_search: behaviour must be unchanged
# ---------------------------------------------------------------------------
def test_symbol_chunk_matches_on_its_own_name(monkeypatch):
    chunks = _corpus(10)
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(["handler_4"], top_k=5)

    assert [r["_idx"] for r in results][0] == "s_4"
    assert results[0]["exact_match_count"] >= 1


def test_symbol_match_is_case_insensitive(monkeypatch):
    chunks = [
        {"text": "x", "is_symbol": True, "symbol_name": "RequestContext", "file": "a.py"}
    ]
    _install_corpus(monkeypatch, chunks)

    assert retriever._exact_symbol_search(["requestcontext"], top_k=5)


def test_non_symbol_chunks_are_still_scanned(monkeypatch):
    chunks = [
        {"text": "value = compute_total(x)", "is_symbol": False, "symbol_name": None,
         "file": "a.py"},
    ]
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(["compute_total"], top_k=5)

    assert len(results) == 1
    assert results[0]["exact_match_count"] == 1


def test_word_boundaries_are_still_enforced(monkeypatch):
    """'get' must not score a chunk containing 'budget', 'target', 'forget'."""
    chunks = [
        {"text": "budget = target + forget", "is_symbol": False, "symbol_name": None,
         "file": "a.py"},
    ]
    _install_corpus(monkeypatch, chunks)

    assert retriever._exact_symbol_search(["get"], top_k=5) == []


def test_repeated_occurrences_still_raise_the_count(monkeypatch):
    chunks = [
        {"text": "parse() ; parse() ; parse()", "is_symbol": False,
         "symbol_name": None, "file": "a.py"},
    ]
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(["parse"], top_k=5)

    assert results[0]["exact_match_count"] == 3


def test_results_are_ranked_by_match_count(monkeypatch):
    chunks = [
        {"text": "parse()", "is_symbol": False, "symbol_name": None, "file": "a.py"},
        {"text": "parse() parse() parse()", "is_symbol": False, "symbol_name": None,
         "file": "b.py"},
        {"text": "parse() parse()", "is_symbol": False, "symbol_name": None,
         "file": "c.py"},
    ]
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(["parse"], top_k=3)

    assert [r["file"] for r in results] == ["b.py", "c.py", "a.py"]


def test_a_chunk_is_returned_once_even_for_duplicate_symbol_candidates(monkeypatch):
    chunks = [{"text": "x", "is_symbol": True, "symbol_name": "Widget", "file": "a.py"}]
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(["Widget", "widget"], top_k=5)

    assert len(results) == 1


def test_empty_symbol_list_short_circuits(monkeypatch):
    called = []

    def _tracking(repo_id=None):
        called.append(repo_id)
        return {"bm25": None, "chunks": []}

    monkeypatch.setattr(retriever, "_get_bm25", _tracking)

    assert retriever._exact_symbol_search([], top_k=5) == []
    assert called == []


def test_symbol_search_respects_metadata_and_repository_filters(monkeypatch):
    chunks = [
        {"text": "x", "is_symbol": True, "symbol_name": "run", "file": "a.py",
         "repository": "alpha"},
        {"text": "x", "is_symbol": True, "symbol_name": "run", "file": "b.py",
         "repository": "beta"},
    ]
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(
        ["run"], top_k=5, repository_names=["beta"]
    )

    assert [r["file"] for r in results] == ["b.py"]


def test_symbol_search_does_not_mutate_the_cached_corpus(monkeypatch):
    chunks = _corpus(6)
    _install_corpus(monkeypatch, chunks)

    retriever._exact_symbol_search(["handler_0", "handler_2"], top_k=5)

    assert all("exact_match_count" not in chunk for chunk in chunks)


def test_symbol_search_honours_top_k(monkeypatch):
    chunks = _corpus(40)
    _install_corpus(monkeypatch, chunks)

    results = retriever._exact_symbol_search(
        [f"handler_{i}" for i in range(0, 40, 2)], top_k=5
    )

    assert len(results) == 5


def test_symbol_search_with_no_index_returns_nothing(monkeypatch):
    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id=None: None)
    assert retriever._exact_symbol_search(["anything"]) == []
