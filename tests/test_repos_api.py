"""Tests for the /repos, /repos/add, and /repos/switch endpoints.

These endpoints were lost once in a merge because no test covered them,
so every behavior below is locked in on purpose: if any of the three
routes disappears or changes shape, the suite fails loudly.
"""

import os

import pytest
from fastapi.testclient import TestClient

import repositories
import main
from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point the registry at a temp dir and neutralize real queueing so
    tests never touch the real output/ directory or start real index jobs."""
    monkeypatch.setattr(repositories, "OUTPUT_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(main, "_queue_index", lambda repo_id: None)
    monkeypatch.setattr(repositories, "current_repo_id", None)


def _register_repo(path):
    """Write a repositories.json containing a single entry for *path*."""
    repo_id = repositories.make_repo_id(str(path))
    repositories.save_repositories(
        [{"id": repo_id, "path": str(path), "name": path.name}]
    )
    return repo_id


def _touch_cache(repo_id):
    """Create an index cache file so the repo reports as already indexed."""
    cache = repositories.cache_path(repo_id)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        f.write("{}")


# ── GET /repos ────────────────────────────────────────────────────────────

def test_list_repos_empty_when_no_registry(client, isolated_state):
    res = client.get("/repos")
    assert res.status_code == 200
    data = res.json()
    assert data["repos"] == []
    assert data["current"] is None


def test_list_repos_returns_registered_repo(client, isolated_state, tmp_path):
    repo_id = _register_repo(tmp_path)
    res = client.get("/repos")
    assert res.status_code == 200
    data = res.json()
    assert len(data["repos"]) == 1
    repo = data["repos"][0]
    assert repo["id"] == repo_id
    assert repo["path"] == str(tmp_path)
    assert repo["name"] == tmp_path.name
    assert repo["indexed"] is False
    assert data["current"] is None


def test_list_repos_reports_indexed_flag(client, isolated_state, tmp_path):
    repo_id = _register_repo(tmp_path)
    _touch_cache(repo_id)
    res = client.get("/repos")
    assert res.status_code == 200
    assert res.json()["repos"][0]["indexed"] is True


def test_list_repos_reflects_current(client, isolated_state, tmp_path):
    repo_id = _register_repo(tmp_path)
    repositories.set_current_repo(repo_id)
    res = client.get("/repos")
    assert res.status_code == 200
    assert res.json()["current"] == repo_id


# ── POST /repos/add ───────────────────────────────────────────────────────

def test_add_repo_success(client, isolated_state, tmp_path):
    res = client.post("/repos/add", json={"path": str(tmp_path)})
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == repositories.make_repo_id(str(tmp_path))
    assert data["indexed"] is False
    assert repositories.load_repositories() == [
        {"id": data["id"], "path": str(tmp_path), "name": tmp_path.name}
    ]


def test_add_repo_requires_path(client, isolated_state):
    res = client.post("/repos/add", json={})
    assert res.status_code == 400


def test_add_repo_rejects_missing_directory(client, isolated_state):
    res = client.post("/repos/add", json={"path": "F:/does/not/exist/xyz"})
    assert res.status_code == 400


def test_add_repo_queues_index_when_not_indexed(client, isolated_state, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_queue_index", lambda repo_id: calls.append(repo_id))
    res = client.post("/repos/add", json={"path": str(tmp_path)})
    assert res.status_code == 200
    assert calls == [res.json()["id"]]


def test_add_repo_does_not_queue_when_indexed(client, isolated_state, tmp_path, monkeypatch):
    repo_id = repositories.make_repo_id(str(tmp_path))
    _touch_cache(repo_id)
    calls = []
    monkeypatch.setattr(main, "_queue_index", lambda rid: calls.append(rid))
    res = client.post("/repos/add", json={"path": str(tmp_path)})
    assert res.status_code == 200
    assert calls == []


# ── POST /repos/switch ────────────────────────────────────────────────────

def test_switch_repo_success(client, isolated_state, tmp_path):
    repo_id = _register_repo(tmp_path)
    res = client.post("/repos/switch", json={"repo_id": repo_id})
    assert res.status_code == 200
    assert res.json() == {"id": repo_id, "indexed": False}
    assert repositories.get_current_repo_id() == repo_id


def test_switch_repo_requires_id(client, isolated_state):
    res = client.post("/repos/switch", json={})
    assert res.status_code == 400


def test_switch_repo_queues_index_when_not_indexed(client, isolated_state, tmp_path, monkeypatch):
    repo_id = _register_repo(tmp_path)
    calls = []
    monkeypatch.setattr(main, "_queue_index", lambda rid: calls.append(rid))
    res = client.post("/repos/switch", json={"repo_id": repo_id})
    assert res.status_code == 200
    assert calls == [repo_id]
