"""Tests for per-repository index metadata resolution (issue #270).

``check_embedding_version()`` called ``get_repository_metadata()`` with no
argument, so it always read ``.index_cache.json`` in the working directory.
Per-repository indexes are written to ``./output/index_cache_<repo_id>.json``
by ``indexer.py``, and ``main.py`` already resolved that path correctly in
``/statistics`` and ``/index/summary`` — ``retriever.py`` never did.

The effect: change the embedding model without re-indexing and the one
diagnostic built to catch it stays silent, because it is reading a file that
either does not exist or belongs to a pre-multi-repo install.
"""

import json
from unittest.mock import MagicMock

import pytest

import retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_metadata_cache():
    retriever.reset_metadata_cache()
    yield
    retriever.reset_metadata_cache()


@pytest.fixture
def write_cache(tmp_path):
    """Write an index-cache JSON and return its path."""

    def _write(metadata, name="index_cache.json", wrapper=None):
        path = tmp_path / name
        body = wrapper if wrapper is not None else {"_metadata": metadata}
        path.write_text(json.dumps(body), encoding="utf-8")
        return str(path)

    return _write


# ---------------------------------------------------------------------------
# metadata_path_for()
# ---------------------------------------------------------------------------
def test_a_registered_repository_resolves_to_its_own_cache():
    path = retriever.metadata_path_for("abc123")
    assert path == retriever.repo_registry.cache_path("abc123")
    assert "abc123" in path


def test_no_repository_falls_back_to_the_legacy_path():
    assert retriever.metadata_path_for(None) == ".index_cache.json"
    assert retriever.metadata_path_for(None) == retriever.LEGACY_METADATA_PATH


def test_two_repositories_resolve_to_different_files():
    assert retriever.metadata_path_for("repoA") != retriever.metadata_path_for("repoB")


def test_metadata_path_matches_what_main_computes():
    """main.py's /statistics uses repositories.cache_path() — stay in step."""
    import repositories

    assert retriever.metadata_path_for("repoA") == repositories.cache_path("repoA")


# ---------------------------------------------------------------------------
# check_embedding_version() reads the right file
# ---------------------------------------------------------------------------
def test_mismatch_is_detected_for_a_registered_repository(monkeypatch, write_cache, caplog):
    repo_cache = write_cache({"embedding_version": "v0-old"})
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: repo_cache)

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")

    assert "Embedding version mismatch" in caplog.text
    assert "v0-old" in caplog.text


def test_no_warning_when_the_versions_agree(monkeypatch, write_cache, caplog):
    repo_cache = write_cache({"embedding_version": retriever.EMBEDDING_VERSION})
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: repo_cache)

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")

    assert "Embedding version mismatch" not in caplog.text


def test_a_legacy_cache_does_not_speak_for_a_registered_repository(monkeypatch, tmp_path, caplog):
    """The exact bug: a stale .index_cache.json must not be consulted.

    The legacy file agrees with config; the repository's own cache does not.
    Reading the wrong one reports a clean bill of health.
    """
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"_metadata": {"embedding_version": retriever.EMBEDDING_VERSION}}),
        encoding="utf-8",
    )
    repo_cache = tmp_path / "index_cache_repoA.json"
    repo_cache.write_text(
        json.dumps({"_metadata": {"embedding_version": "v0-stale"}}), encoding="utf-8"
    )

    monkeypatch.setattr(retriever, "LEGACY_METADATA_PATH", str(legacy))
    monkeypatch.setattr(
        retriever.repo_registry, "cache_path", lambda repo_id: str(repo_cache)
    )

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")

    assert "v0-stale" in caplog.text


def test_missing_metadata_is_silent(monkeypatch, caplog):
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: "/nope/missing.json")

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")

    assert "Embedding version mismatch" not in caplog.text


def test_metadata_without_a_version_is_silent(monkeypatch, write_cache, caplog):
    repo_cache = write_cache({"indexed_file_count": 12})
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: repo_cache)

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")

    assert "Embedding version mismatch" not in caplog.text


# ---------------------------------------------------------------------------
# Warn once, not once per query
# ---------------------------------------------------------------------------
def test_the_same_mismatch_warns_once(monkeypatch, write_cache, caplog):
    repo_cache = write_cache({"embedding_version": "v0-old"})
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: repo_cache)

    with caplog.at_level("WARNING"):
        for _ in range(5):
            retriever.check_embedding_version("repoA")

    assert caplog.text.count("Embedding version mismatch") == 1


def test_a_different_repository_warns_separately(monkeypatch, write_cache, caplog):
    repo_cache = write_cache({"embedding_version": "v0-old"})
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: repo_cache)

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")
        retriever.check_embedding_version("repoB")

    assert caplog.text.count("Embedding version mismatch") == 2


def test_resetting_the_cache_re_arms_the_warning(monkeypatch, write_cache, caplog):
    repo_cache = write_cache({"embedding_version": "v0-old"})
    monkeypatch.setattr(retriever, "metadata_path_for", lambda repo_id: repo_cache)

    with caplog.at_level("WARNING"):
        retriever.check_embedding_version("repoA")
        retriever.reset_metadata_cache()
        retriever.check_embedding_version("repoA")

    assert caplog.text.count("Embedding version mismatch") == 2


# ---------------------------------------------------------------------------
# get_repository_metadata(): parsing, caching, invalidation
# ---------------------------------------------------------------------------
def test_metadata_block_is_returned(write_cache):
    path = write_cache({"repository_name": "devwhisper", "indexed_file_count": 42})

    metadata = retriever.get_repository_metadata(path)

    assert metadata["repository_name"] == "devwhisper"
    assert metadata["indexed_file_count"] == 42


def test_missing_file_returns_empty_dict():
    assert retriever.get_repository_metadata("/nope/missing.json") == {}


def test_corrupted_json_returns_empty_dict(tmp_path, caplog):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert retriever.get_repository_metadata(str(path)) == {}

    assert "Corrupted repository metadata" in caplog.text


def test_a_file_without_a_metadata_block_returns_empty_dict(write_cache):
    path = write_cache(None, wrapper={"some_file.py": {"hash": "abc"}})
    assert retriever.get_repository_metadata(path) == {}


def test_a_non_dict_metadata_block_returns_empty_dict(write_cache):
    path = write_cache(None, wrapper={"_metadata": ["not", "a", "dict"]})
    assert retriever.get_repository_metadata(path) == {}


def test_a_second_read_does_not_reparse_the_file(write_cache, monkeypatch):
    path = write_cache({"indexed_file_count": 7})
    retriever.get_repository_metadata(path)

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the file was re-parsed despite being unchanged")

    monkeypatch.setattr(json, "load", explode)

    assert retriever.get_repository_metadata(path)["indexed_file_count"] == 7


def test_a_rewritten_file_is_picked_up(tmp_path):
    path = tmp_path / "index_cache.json"
    path.write_text(json.dumps({"_metadata": {"indexed_file_count": 1}}), encoding="utf-8")
    assert retriever.get_repository_metadata(str(path))["indexed_file_count"] == 1

    path.write_text(
        json.dumps({"_metadata": {"indexed_file_count": 99, "padding": "x" * 50}}),
        encoding="utf-8",
    )

    assert retriever.get_repository_metadata(str(path))["indexed_file_count"] == 99


def test_a_file_that_appears_later_is_picked_up(tmp_path):
    """A repository indexed after its first query must start reporting."""
    path = tmp_path / "index_cache.json"
    assert retriever.get_repository_metadata(str(path)) == {}

    path.write_text(json.dumps({"_metadata": {"indexed_file_count": 3}}), encoding="utf-8")

    assert retriever.get_repository_metadata(str(path))["indexed_file_count"] == 3


def test_callers_cannot_mutate_the_cache(write_cache):
    path = write_cache({"indexed_file_count": 5})

    first = retriever.get_repository_metadata(path)
    first["indexed_file_count"] = 999
    first["injected"] = True

    second = retriever.get_repository_metadata(path)
    assert second["indexed_file_count"] == 5
    assert "injected" not in second


def test_two_repositories_are_cached_independently(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"_metadata": {"repository_name": "alpha"}}), encoding="utf-8")
    b.write_text(json.dumps({"_metadata": {"repository_name": "beta"}}), encoding="utf-8")

    assert retriever.get_repository_metadata(str(a))["repository_name"] == "alpha"
    assert retriever.get_repository_metadata(str(b))["repository_name"] == "beta"
    assert retriever.get_repository_metadata(str(a))["repository_name"] == "alpha"


# ---------------------------------------------------------------------------
# retrieve() passes the repository through
# ---------------------------------------------------------------------------
def test_retrieve_checks_the_active_repository(monkeypatch):
    seen = []
    monkeypatch.setattr(retriever, "check_embedding_version", lambda rid=None: seen.append(rid))

    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    monkeypatch.setattr(retriever, "client", client)
    embedder = MagicMock()
    embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setattr(retriever, "embedder", embedder)
    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id: None)

    retriever.retrieve("a question", repo_id="repoA")

    assert seen == ["repoA"]
