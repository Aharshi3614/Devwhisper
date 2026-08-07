from repositories import *
import repositories
from config import (OUTPUT_DIRECTORY,
                    QDRANT_COLLECTION_NAME)
import pytest
import json

def test_make_repo_id():
    hash_repo_id = make_repo_id("/Devwhisper/output")
    with pytest.raises(ValueError):
        make_repo_id(None)
    assert hash_repo_id is not None

def test_collection_name():
    repo_name = collection_name("opencode")
    with pytest.raises(ValueError):
        collection_name(None)
    assert repo_name == f"{QDRANT_COLLECTION_NAME}_opencode"

def test_bm25_path():
    repo_name = bm25_path("opencode")
    with pytest.raises(ValueError):
        bm25_path(None)
    assert repo_name == f"{OUTPUT_DIRECTORY}/bm_index_opencode.pkl"

def test_cache_path():
    repo_name = cache_path("opencode")
    with pytest.raises(ValueError):
        cache_path(None)
    assert repo_name == f"{OUTPUT_DIRECTORY}/index_cache_opencode.json"


@pytest.fixture
def tmp_output(tmp_path, monkeypatch):
    """Point OUTPUT_DIRECTORY at a temp dir so tests never touch real files."""
    monkeypatch.setattr(repositories, "OUTPUT_DIRECTORY", str(tmp_path))
    return tmp_path


def test_load_repositories_returns_empty_when_missing(tmp_output):
    assert load_repositories() == []


def test_load_repositories_returns_list_from_file(tmp_output):
    (tmp_output / "repositories.json").write_text(
        json.dumps([{"id": "abc", "path": "F:/x", "name": "x"}])
    )
    assert load_repositories() == [{"id": "abc", "path": "F:/x", "name": "x"}]


def test_save_then_load_round_trip(tmp_output):
    repos = [{"id": "abc", "path": "F:/x", "name": "x"}]
    save_repositories(repos)
    assert load_repositories() == repos


def test_save_creates_output_directory(tmp_output):
    save_repositories([])
    assert (tmp_output / "repositories.json").exists()


# ── add_repository ────────────────────────────────────────────────────────

def test_add_repository_creates_entry(tmp_output, tmp_path):
    repo_id = add_repository(str(tmp_path))
    repos = load_repositories()
    assert len(repos) == 1
    assert repos[0]["id"] == repo_id
    assert repos[0]["path"] == str(tmp_path)


def test_add_repository_is_idempotent(tmp_output, tmp_path):
    """Adding the same path twice yields the same id and no duplicate entry."""
    rid1 = add_repository(str(tmp_path))
    rid2 = add_repository(str(tmp_path))
    assert rid1 == rid2
    assert len(load_repositories()) == 1


# ── get_current_repo_name ─────────────────────────────────────────────────

def test_get_current_repo_name_returns_none_when_no_active(tmp_output, monkeypatch):
    """With no active repository, the display name is None."""
    monkeypatch.setattr(repositories, "current_repo_id", None)
    assert get_current_repo_name() is None


def test_get_current_repo_name_returns_basename(tmp_output, monkeypatch):
    """The active repository's name is its path basename (matches indexer tag)."""
    monkeypatch.setattr(repositories, "current_repo_id", "abc")
    save_repositories([{"id": "abc", "path": "F:/some/path/myproj", "name": "myproj"}])
    assert get_current_repo_name() == "myproj"


def test_get_current_repo_name_returns_none_for_unknown_id(tmp_output, monkeypatch):
    """An id that is not in the registry has no display name."""
    monkeypatch.setattr(repositories, "current_repo_id", "ghost")
    save_repositories([{"id": "abc", "path": "F:/x", "name": "x"}])
    assert get_current_repo_name() is None

