"""
repositories.py — Repository registry and per-repository index paths for DevWhisper.

This module manages the list of repositories DevWhisper can search and derives
each repository's unique index-artifact names (Qdrant collection, BM25 file,
index cache file). It also tracks which repository is currently active, so the
retrieval and change-detection code knows what to query.

Each repository is identified by a stable id hashed from its on-disk path, so
the same path always maps to the same artifacts and different paths never
collide.

Key components:
  - make_repo_id(): Hashes a repository path into a stable short id.
  - collection_name(): Qdrant collection name for a repository's vectors.
  - bm25_path(): Local file path of a repository's BM25 keyword index.
  - cache_path(): Local file path of a repository's index cache.
  - load_repositories(): Loads repositories from disk.
  - save_repositories(): Saves repositories to disk.
  - set_current_repo(): Sets current repository to this repository.
  - get_current_repo_id(): Gets current repository.
"""
import os

from logger import logger

from config import QDRANT_COLLECTION_NAME, OUTPUT_DIRECTORY
import hashlib
import json

current_repo_id: str | None = None

def make_repo_id(path: str) -> str:
    """A stable short id hashed from the repository's path"""
    if path is None:
        raise ValueError("hash cannot be None")
    return hashlib.sha1(path.encode()).hexdigest()[:12]

def collection_name(repo_id: str) -> str:
    """Qdrant collection for a repository's vectors"""
    if repo_id is None:
        raise ValueError("repo_id cannot be None")
    return f"{QDRANT_COLLECTION_NAME}_{repo_id}"

def  bm25_path(repo_id: str) -> str:
    """the path of bm25 output files"""
    if repo_id is None:
        raise ValueError("repo_id cannot be None")
    return f"{OUTPUT_DIRECTORY}/bm_index_{repo_id}.pkl"

def cache_path(repo_id: str) -> str:
    """the path of cache output files"""
    if repo_id is None:
        raise ValueError("repo_id cannot be None")
    return f"{OUTPUT_DIRECTORY}/index_cache_{repo_id}.json"

def load_repositories() -> list:
    """load all repositories from ./{OUTPUT_DIRECTORY}/repositories.json"""
    if not os.path.exists(f"{OUTPUT_DIRECTORY}/repositories.json"):
        logger.debug("repositories.json not found.")
        return []
    try:
        with open (f"{OUTPUT_DIRECTORY}/repositories.json", "r") as f:
            repositories_list = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load repositories: {e}")
        return []
    return repositories_list

def save_repositories(repos: list) -> None:
    """Write the repositories list to ./{OUTPUT_DIRECTORY}/repositories.json"""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with open(f"{OUTPUT_DIRECTORY}/repositories.json", "w") as f:
        json.dump(repos, f, indent=4)

def add_repository(path: str) -> str:
    """Register a repository path and return its repo id."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} not found.")
    repo_id = make_repo_id(path)
    repos = load_repositories()
    for existing in repos:
        if existing["path"] == path:
            return repo_id
    repos.append({"id": repo_id, "path": path, "name": os.path.basename(path)})
    save_repositories(repos)
    return repo_id

def set_current_repo(repo_id: str) -> None:
    """Set which repository is currently active."""
    global current_repo_id
    current_repo_id = repo_id

def get_current_repo_id() -> str | None:
    """Return the id of the currently active repository(or None)."""
    return current_repo_id

def get_repo_path(repo_id: str) -> str | None:
    """Return the on-disk path registered for *repo_id*, or None if unknown."""
    if repo_id is None:
        raise ValueError("repo_id cannot be None")
    for repo in load_repositories():
        if repo["id"] == repo_id: return repo["path"]
    return None

def get_current_repo_name() -> str | None:
    """Return the display name of the currently active repository (or None).

    Mirrors how the indexer stamps the ``repository`` tag (path basename),
    so name filtering in retrieve() lines up with the tag written at index time.
    """
    repo_id = get_current_repo_id()
    if repo_id is None:
        return None
    path = get_repo_path(repo_id)
    if path is None:
        return None
    return os.path.basename(os.path.normpath(path))