"""BM25 index reload tests (issue #259).

`retriever._get_bm25()` loaded a repository's BM25 pickle once and kept it in
the module-level `_bm25_data` dict forever:

    if repo_id not in _bm25_data:      # only a miss ever triggered a load
        _bm25_data[repo_id] = pickle.load(f)

`index_directory()` rewrites that pickle at the end of every run — in the
same process that holds the cached copy — so after a re-index the server kept
scoring keyword and symbol search against the previous snapshot until it was
restarted. Dense vector search hits Qdrant live and did see the new code, so
the two halves of hybrid retrieval drifted apart, and RRF kept boosting the
stale chunks because they appeared in two of the three ranked lists.

A failed load was cached just as permanently: a repository with no BM25 index
at first query stayed "no keyword search" forever, even after indexing built
one.
"""

import pickle
from pathlib import Path

import pytest

import retriever


class FakeBM25:
    """Stand-in for BM25Okapi that survives a pickle round-trip.

    conftest replaces `rank_bm25` with a MagicMock, so the real class cannot
    be pickled here. Only the interface `_keyword_search` uses is needed.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        return [1.0] * len(self.corpus)


def _write_index(path: Path, files: list[str]) -> None:
    """Write a BM25 pickle in the shape the indexer produces."""
    chunks = [{"text": f"code in {name}", "file": name} for name in files]
    with open(path, "wb") as f:
        pickle.dump({"bm25": FakeBM25([c["text"] for c in chunks]), "chunks": chunks}, f)


def _files(payload) -> list[str]:
    return [chunk["file"] for chunk in payload["chunks"]]


@pytest.fixture
def index_path(tmp_path, monkeypatch):
    """Point retriever at a throwaway BM25 pickle and start from a clean cache."""
    path = tmp_path / "bm_index_repoX.pkl"
    monkeypatch.setattr(retriever.repo_registry, "bm25_path", lambda repo_id: str(path))
    retriever.clear_bm25_cache()
    yield path
    retriever.clear_bm25_cache()


# ---------------------------------------------------------------------------
# Reload after re-indexing
# ---------------------------------------------------------------------------

def test_rewritten_index_is_picked_up(index_path):
    """The regression: a re-index used to be invisible until restart."""
    _write_index(index_path, ["old.py"])
    assert _files(retriever._get_bm25("repoX")) == ["old.py"]

    _write_index(index_path, ["new.py"])  # simulates index_directory() finishing

    assert _files(retriever._get_bm25("repoX")) == ["new.py"]


def test_reload_survives_a_same_second_rewrite(index_path):
    """Two writes inside one clock tick still differ by size."""
    _write_index(index_path, ["a.py"])
    retriever._get_bm25("repoX")

    _write_index(index_path, ["a.py", "b.py", "c.py"])

    assert _files(retriever._get_bm25("repoX")) == ["a.py", "b.py", "c.py"]


def test_unchanged_index_is_not_re_read(index_path, monkeypatch):
    """Steady state must cost a stat(), not a pickle.load(), per query."""
    _write_index(index_path, ["a.py"])
    retriever._get_bm25("repoX")

    loads = []
    real_load = pickle.load

    def counting_load(f):
        loads.append(1)
        return real_load(f)

    monkeypatch.setattr(retriever.pickle, "load", counting_load)

    for _ in range(5):
        retriever._get_bm25("repoX")

    assert loads == [], "an unchanged index should not be deserialized again"


def test_repeated_calls_return_the_same_payload(index_path):
    _write_index(index_path, ["a.py"])

    first = retriever._get_bm25("repoX")
    second = retriever._get_bm25("repoX")

    assert first is second, "cached payload should be reused, not rebuilt"


# ---------------------------------------------------------------------------
# Missing index
# ---------------------------------------------------------------------------

def test_missing_index_returns_none(index_path):
    assert retriever._get_bm25("repoX") is None


def test_index_created_after_the_first_query_is_picked_up(index_path):
    """A repository indexed after its first query used to stay broken.

    The old code cached the FileNotFoundError result under the repo id, so
    keyword and symbol search stayed disabled for the life of the process.
    """
    assert retriever._get_bm25("repoX") is None

    _write_index(index_path, ["fresh.py"])

    payload = retriever._get_bm25("repoX")
    assert payload is not None
    assert _files(payload) == ["fresh.py"]


def test_deleted_index_returns_none_again(index_path):
    _write_index(index_path, ["a.py"])
    assert retriever._get_bm25("repoX") is not None

    index_path.unlink()

    assert retriever._get_bm25("repoX") is None


def test_corrupt_index_returns_none_without_raising(index_path):
    index_path.write_bytes(b"this is not a pickle")

    assert retriever._get_bm25("repoX") is None


def test_corrupt_index_recovers_once_rewritten(index_path):
    index_path.write_bytes(b"this is not a pickle")
    assert retriever._get_bm25("repoX") is None

    _write_index(index_path, ["repaired.py"])

    assert _files(retriever._get_bm25("repoX")) == ["repaired.py"]


# ---------------------------------------------------------------------------
# Explicit invalidation
# ---------------------------------------------------------------------------

def test_invalidate_forces_a_reload(index_path, monkeypatch):
    _write_index(index_path, ["a.py"])
    retriever._get_bm25("repoX")

    loads = []
    real_load = pickle.load
    monkeypatch.setattr(
        retriever.pickle, "load", lambda f: (loads.append(1), real_load(f))[1]
    )

    retriever.invalidate_bm25_cache("repoX")
    retriever._get_bm25("repoX")

    assert len(loads) == 1


def test_clear_drops_every_repository(tmp_path, monkeypatch):
    paths = {
        "repoA": tmp_path / "a.pkl",
        "repoB": tmp_path / "b.pkl",
    }
    monkeypatch.setattr(
        retriever.repo_registry, "bm25_path", lambda repo_id: str(paths[repo_id])
    )
    retriever.clear_bm25_cache()

    for repo_id, path in paths.items():
        _write_index(path, [f"{repo_id}.py"])
        retriever._get_bm25(repo_id)

    assert len(retriever._bm25_data) == 2

    retriever.clear_bm25_cache()

    assert retriever._bm25_data == {}


# ---------------------------------------------------------------------------
# Repository isolation
# ---------------------------------------------------------------------------

def test_repositories_do_not_share_a_cache_entry(tmp_path, monkeypatch):
    paths = {"repoA": tmp_path / "a.pkl", "repoB": tmp_path / "b.pkl"}
    monkeypatch.setattr(
        retriever.repo_registry, "bm25_path", lambda repo_id: str(paths[repo_id])
    )
    retriever.clear_bm25_cache()

    _write_index(paths["repoA"], ["from_a.py"])
    _write_index(paths["repoB"], ["from_b.py"])

    assert _files(retriever._get_bm25("repoA")) == ["from_a.py"]
    assert _files(retriever._get_bm25("repoB")) == ["from_b.py"]

    _write_index(paths["repoA"], ["from_a_v2.py"])

    assert _files(retriever._get_bm25("repoA")) == ["from_a_v2.py"]
    assert _files(retriever._get_bm25("repoB")) == ["from_b.py"], "repoB untouched"

    retriever.clear_bm25_cache()


def test_default_index_path_is_used_when_repo_id_is_none(tmp_path, monkeypatch):
    path = tmp_path / "shared.pkl"
    monkeypatch.setattr(retriever, "BM25_INDEX_PATH", str(path))
    retriever.clear_bm25_cache()

    _write_index(path, ["shared.py"])

    assert _files(retriever._get_bm25(None)) == ["shared.py"]

    retriever.clear_bm25_cache()


# ---------------------------------------------------------------------------
# Reaching the search paths
# ---------------------------------------------------------------------------

def test_keyword_search_sees_the_reindexed_content(index_path):
    """The user-visible symptom, one level up from _get_bm25()."""
    _write_index(index_path, ["old.py"])
    assert [c["file"] for c in retriever._keyword_search("code", repo_id="repoX")] == ["old.py"]

    _write_index(index_path, ["new.py"])

    assert [c["file"] for c in retriever._keyword_search("code", repo_id="repoX")] == ["new.py"]


def test_symbol_search_sees_the_reindexed_content(index_path):
    def write_symbols(path, symbol, file_name):
        chunks = [
            {
                "text": f"def {symbol}(): pass",
                "file": file_name,
                "symbol_name": symbol,
                "is_symbol": True,
            }
        ]
        with open(path, "wb") as f:
            pickle.dump({"bm25": FakeBM25(["x"]), "chunks": chunks}, f)

    write_symbols(index_path, "old_function", "old.py")
    assert retriever._exact_symbol_search(["new_function"], repo_id="repoX") == []

    write_symbols(index_path, "new_function", "new.py")

    matches = retriever._exact_symbol_search(["new_function"], repo_id="repoX")
    assert [m["file"] for m in matches] == ["new.py"]
