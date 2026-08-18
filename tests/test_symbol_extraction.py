"""Tests for symbol extraction and exact symbol matching (issue #268).

``_extract_symbols()`` used to classify any capitalised word as a class name
(``\\b([A-Z][a-zA-Z0-9]+)\\b``). Since questions start with a capital letter,
"How", "What" and "Where" came back as code symbols on nearly every query,
and ``_exact_symbol_search()`` then ranked chunks by how many times those
words appeared in their source text — effectively sorting by prose volume.

These tests pin down what counts as a symbol, what does not, and that the
substring branch of the symbol search matches on word boundaries.
"""

from unittest.mock import MagicMock

import pytest

import retriever


# ---------------------------------------------------------------------------
# Question words are not symbols
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, unwanted",
    [
        ("How does the retriever work?", "How"),
        ("What is the collection name?", "What"),
        ("Where is the index stored?", "Where"),
        ("Why does indexing skip files?", "Why"),
        ("When is the cache invalidated?", "When"),
        ("Which files are indexed?", "Which"),
        ("Who calls this function?", "Who"),
        ("Does the webhook stream?", "Does"),
        ("Can I re-index a repository?", "Can"),
        ("Should the answer be cached?", "Should"),
        ("The pipeline runs in stages.", "The"),
        ("Is the index stale?", "Is"),
    ],
)
def test_sentence_initial_words_are_not_symbols(query, unwanted):
    assert unwanted not in retriever._extract_symbols(query)


def test_a_plain_question_yields_no_symbols_at_all():
    """Nothing in this sentence is code, so nothing should be extracted."""
    assert retriever._extract_symbols("How does the search work here?") == []


def test_all_caps_question_words_are_filtered_by_the_stopword_list():
    """Shouted questions match the acronym rule, so the stopwords catch them."""
    symbols = retriever._extract_symbols("WHERE IS THE PARSER")
    assert "WHERE" not in symbols
    assert "IS" not in symbols
    assert "THE" not in symbols


def test_stopwords_are_matched_case_insensitively():
    """SHOULD/THIS/BE are dropped; CACHED is kept — it could be an acronym.

    An unknown all-caps word is genuinely ambiguous, so the stopword list is
    the only thing separating scaffolding from a name here.
    """
    symbols = retriever._extract_symbols("SHOULD THIS BE CACHED")
    assert symbols == ["CACHED"]


# ---------------------------------------------------------------------------
# Real symbols still come through
# ---------------------------------------------------------------------------
def test_call_syntax_is_still_extracted():
    assert "retrieve" in retriever._extract_symbols("What does the retrieve() function do?")


def test_call_syntax_wins_over_the_stopword_list_is_not_assumed():
    """"get" is a stopword, so `get()` is dropped — documented, not accidental."""
    # A bare "get" carries no information: it appears in prose constantly and
    # matches dozens of chunks. Users asking about a specific getter write its
    # qualified name, which is preserved below.
    assert "get" not in retriever._extract_symbols("what does get() return?")
    assert "cache_get" in retriever._extract_symbols("what does cache_get() return?")


def test_camel_case_class_names_are_extracted():
    assert "DataProcessor" in retriever._extract_symbols("Where is the DataProcessor class?")


def test_camel_case_with_a_leading_lowercase_segment_is_extracted():
    assert "RequestContext" in retriever._extract_symbols("Explain RequestContext please")


def test_acronyms_are_extracted():
    symbols = retriever._extract_symbols("Where does the BM25 index live?")
    assert "BM25" in symbols


def test_short_acronyms_are_extracted():
    assert "RRF" in retriever._extract_symbols("How is the RRF score computed?")


def test_mixed_case_product_names_are_extracted():
    assert "DevWhisper" in retriever._extract_symbols("What does DevWhisper index?")


def test_snake_case_identifiers_are_extracted():
    symbols = retriever._extract_symbols("Where is make_repo_id defined?")
    assert "make_repo_id" in symbols


def test_snake_case_with_digits_is_extracted():
    assert "bm25_path" in retriever._extract_symbols("what does bm25_path return")


# ---------------------------------------------------------------------------
# Single capitalised words are a deliberate exclusion
# ---------------------------------------------------------------------------
def test_single_capitalised_words_are_not_symbols():
    """"Qdrant" is shaped exactly like "Where" — we cannot tell them apart.

    Dense and BM25 search both still see the plain token, so nothing is lost
    for the query as a whole; only the exact-symbol list is kept clean.
    """
    symbols = retriever._extract_symbols("What is the Qdrant collection name?")
    assert "Qdrant" not in symbols
    assert "What" not in symbols


# ---------------------------------------------------------------------------
# Ordering and de-duplication
# ---------------------------------------------------------------------------
def test_symbols_are_returned_in_a_deterministic_order():
    """This used to be `list(set(...))`, whose order varies between runs."""
    query = "Does make_repo_id() call DataProcessor or cache_path?"
    assert retriever._extract_symbols(query) == retriever._extract_symbols(query)


def test_a_symbol_mentioned_twice_appears_once():
    symbols = retriever._extract_symbols("does retrieve() call retrieve() again?")
    assert symbols.count("retrieve") == 1


def test_a_symbol_matching_two_heuristics_appears_once():
    """`make_repo_id()` is both call syntax and snake_case."""
    symbols = retriever._extract_symbols("what does make_repo_id() do?")
    assert symbols.count("make_repo_id") == 1


def test_empty_query_yields_no_symbols():
    assert retriever._extract_symbols("") == []


# ---------------------------------------------------------------------------
# _exact_symbol_search: word-boundary matching
# ---------------------------------------------------------------------------
def _corpus(monkeypatch, chunks):
    monkeypatch.setattr(
        retriever,
        "_get_bm25",
        lambda repo_id: {"bm25": MagicMock(), "chunks": chunks},
    )


def test_substring_matches_inside_larger_words_are_not_counted(monkeypatch):
    """Searching for "get" must not score a chunk full of "budget"."""
    _corpus(monkeypatch, [
        {"text": "budget = target - forget_this", "is_symbol": False},
    ])

    assert retriever._exact_symbol_search(["get"], top_k=5) == []


def test_a_genuine_whole_word_match_is_still_counted(monkeypatch):
    _corpus(monkeypatch, [
        {"text": "value = get(key)", "is_symbol": False},
    ])

    results = retriever._exact_symbol_search(["get"], top_k=5)
    assert len(results) == 1
    assert results[0]["exact_match_count"] == 1


def test_snake_case_names_are_matched_whole(monkeypatch):
    """`cache_path` must not match `cache_path_for`."""
    _corpus(monkeypatch, [
        {"text": "def cache_path_for(repo): pass", "is_symbol": False},
        {"text": "p = cache_path(repo_id)", "is_symbol": False},
    ])

    results = retriever._exact_symbol_search(["cache_path"], top_k=5)
    assert len(results) == 1
    assert "cache_path(repo_id)" in results[0]["text"]


def test_repeated_whole_word_occurrences_are_all_counted(monkeypatch):
    _corpus(monkeypatch, [
        {"text": "get(a); get(b); get(c)", "is_symbol": False},
    ])

    results = retriever._exact_symbol_search(["get"], top_k=5)
    assert results[0]["exact_match_count"] == 3


def test_symbols_with_regex_metacharacters_do_not_explode(monkeypatch):
    """Candidates are escaped before compiling — a stray "." must be literal."""
    _corpus(monkeypatch, [
        {"text": "self.value = 1", "is_symbol": False},
    ])

    # Should neither raise nor match "selfXvalue"-style text.
    assert retriever._exact_symbol_search(["a.c"], top_k=5) == []


def test_metadata_symbol_matching_is_unaffected(monkeypatch):
    """The exact `symbol_name` branch keeps its existing equality semantics."""
    _corpus(monkeypatch, [
        {"text": "def preprocess(d): pass", "symbol_name": "preprocess", "is_symbol": True},
        {"text": "def preprocess_data(x): pass", "symbol_name": "preprocess_data", "is_symbol": True},
    ])

    results = retriever._exact_symbol_search(["preprocess"], top_k=5)
    names = [r.get("symbol_name") for r in results]
    assert names == ["preprocess"]


def test_no_symbols_returns_empty_without_touching_the_corpus(monkeypatch):
    def explode(repo_id):  # pragma: no cover - must not run
        raise AssertionError("the corpus should not be loaded for an empty symbol list")

    monkeypatch.setattr(retriever, "_get_bm25", explode)
    assert retriever._exact_symbol_search([], top_k=5) == []


# ---------------------------------------------------------------------------
# The two fixes together
# ---------------------------------------------------------------------------
def test_a_prose_question_no_longer_ranks_chunks_by_wordiness(monkeypatch):
    """End to end: "How does the retriever work?" must match nothing by prose."""
    _corpus(monkeypatch, [
        {
            "text": "# This shows how the shower of results is handled, and how "
                    "the code below shows how that works.",
            "is_symbol": False,
        },
        {"text": "def retrieve(): pass", "symbol_name": "retrieve", "is_symbol": True},
    ])

    symbols = retriever._extract_symbols("How does the retriever work?")
    results = retriever._exact_symbol_search(symbols, top_k=5)

    assert symbols == []
    assert results == []
