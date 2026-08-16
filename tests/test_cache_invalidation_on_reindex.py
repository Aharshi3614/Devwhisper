"""Tests for response-cache invalidation after indexing (issue #269).

``cache.invalidate_repo()`` was written for this and documented as
"e.g. after re-index", but nothing called it. A repository could be
re-indexed and every previously asked question kept returning the answer
generated from the *previous* snapshot of the code — with a "Sources used:"
footer citing files the repository might no longer contain.

These tests drive the real ``queue_worker()`` loop with a stubbed
``index_directory``, so the wiring is exercised end to end rather than
asserted against a mock of itself.
"""

import queue

import pytest

import cache
import main
import retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_cache():
    """Start and finish each test with an empty response cache."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def scoped(monkeypatch):
    """Pin the active repository so cache keys are predictable."""

    def _scoped(repo_id):
        monkeypatch.setattr(cache.repositories, "get_current_repo_id", lambda: repo_id)
        return repo_id

    return _scoped


@pytest.fixture
def run_job(monkeypatch):
    """Run a single job through the real queue_worker() and return it.

    A ``None`` sentinel is queued behind the job so the worker loop exits
    instead of blocking forever.
    """

    def _run_job(job, index_directory=None, repo_path="/tmp/repo"):
        work = queue.Queue()
        work.put(job)
        work.put(None)

        calls = []

        def default_index_directory(directory, repo_id=None, dry_run=False):
            calls.append({"directory": directory, "repo_id": repo_id, "dry_run": dry_run})

        monkeypatch.setattr(main, "indexing_queue", work)
        monkeypatch.setattr(
            main, "index_directory", index_directory or default_index_directory
        )
        monkeypatch.setattr(main.repositories, "get_repo_path", lambda rid: repo_path)

        main.queue_worker()
        return calls

    return _run_job


def _job(**overrides):
    job = {
        "id": "job-1",
        "type": "reindex",
        "name": "Manual Re-index",
        "repo_id": "repoA",
        "dry_run": False,
        "status": "pending",
        "percent": 0,
        "message": "",
        "created_at": 0.0,
        "started_at": 0.0,
        "finished_at": 0.0,
        "error": None,
    }
    job.update(overrides)
    return job


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------
def test_reindexing_drops_that_repository_s_cached_answers(scoped, run_job):
    scoped("repoA")
    cache.put("what does retrieve return", "the old answer")
    assert cache.get("what does retrieve return") == "the old answer"

    run_job(_job(repo_id="repoA"))

    assert cache.get("what does retrieve return") is None


def test_a_stale_answer_is_not_resurrected_by_similarity_matching(scoped, run_job):
    """Near-duplicate matching must not find the entry either."""
    scoped("repoA")
    cache.put("what does the retrieve function return", "the old answer")

    run_job(_job(repo_id="repoA"))

    # Same tokens, different order — well above the 0.70 threshold.
    assert cache.get("what does the retrieve function return exactly") is None


def test_other_repositories_keep_their_cached_answers(monkeypatch, run_job):
    """Re-indexing repoA must not evict repoB's answers."""
    monkeypatch.setattr(cache.repositories, "get_current_repo_id", lambda: "repoB")
    cache.put("a question about repo b", "repo b answer")

    monkeypatch.setattr(cache.repositories, "get_current_repo_id", lambda: "repoA")
    cache.put("a question about repo a", "repo a answer")

    run_job(_job(repo_id="repoA"))

    assert cache.get("a question about repo a") is None

    monkeypatch.setattr(cache.repositories, "get_current_repo_id", lambda: "repoB")
    assert cache.get("a question about repo b") == "repo b answer"


def test_upload_jobs_also_invalidate(scoped, run_job, tmp_path, monkeypatch):
    """An uploaded ZIP replaces the codebase, so its answers are stale too."""
    scoped("repoA")
    cache.put("what does this project do", "answer from the previous upload")

    # The upload branch extracts a ZIP; stub the filesystem work away and
    # only keep the job type, which is what selects the branch.
    monkeypatch.setattr(main.os.path, "exists", lambda p: False)
    monkeypatch.setattr(main.os, "makedirs", lambda *a, **kw: None)

    class _FakeZip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extractall(self, target):
            pass

    monkeypatch.setattr(main.zipfile, "ZipFile", _FakeZip)

    run_job(_job(type="upload", temp_zip_path=str(tmp_path / "x.zip")))

    assert cache.get("what does this project do") is None


# ---------------------------------------------------------------------------
# Dry runs are exempt
# ---------------------------------------------------------------------------
def test_a_dry_run_keeps_the_cache_warm(scoped, run_job):
    """A dry run uploads no vectors, so throwing away the cache is pure loss."""
    scoped("repoA")
    cache.put("what does retrieve return", "still valid")

    run_job(_job(id="job-dry", type="dry_run", dry_run=True))

    assert cache.get("what does retrieve return") == "still valid"


def test_a_dry_run_still_reaches_the_indexer(scoped, run_job):
    """Exempting dry runs from invalidation must not skip the job itself."""
    scoped("repoA")
    calls = run_job(_job(id="job-dry", type="dry_run", dry_run=True))

    assert len(calls) == 1
    assert calls[0]["dry_run"] is True


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------
def test_a_failed_job_still_invalidates(scoped, run_job):
    """index_directory() upserts in batches; a partial run still changed things."""
    scoped("repoA")
    cache.put("what does retrieve return", "answer from before the failed run")

    def boom(directory, repo_id=None, dry_run=False):
        raise RuntimeError("qdrant unreachable")

    run_job(_job(repo_id="repoA"), index_directory=boom)

    assert cache.get("what does retrieve return") is None


def test_the_job_is_still_marked_failed(scoped, run_job):
    scoped("repoA")
    job = _job(repo_id="repoA")

    def boom(directory, repo_id=None, dry_run=False):
        raise RuntimeError("qdrant unreachable")

    run_job(job, index_directory=boom)

    assert job["status"] == "failed"
    assert "qdrant unreachable" in job["error"]


def test_a_successful_job_is_still_marked_completed(scoped, run_job):
    scoped("repoA")
    job = _job(repo_id="repoA")

    run_job(job)

    assert job["status"] == "completed"
    assert job["percent"] == 100


# ---------------------------------------------------------------------------
# invalidate_repo_caches() directly
# ---------------------------------------------------------------------------
def test_invalidate_repo_caches_drops_the_bm25_payload(monkeypatch):
    retriever._bm25_data["repoA"] = ((1, 2), {"bm25": None, "chunks": []})
    retriever._bm25_data["repoB"] = ((3, 4), {"bm25": None, "chunks": []})

    main.invalidate_repo_caches("repoA")

    assert "repoA" not in retriever._bm25_data
    assert "repoB" in retriever._bm25_data
    retriever._bm25_data.clear()


def test_invalidate_repo_caches_survives_a_broken_response_cache(monkeypatch):
    """A cache failure must not take the worker thread down."""
    def boom(repo_id):
        raise RuntimeError("cache exploded")

    monkeypatch.setattr(main, "cache_invalidate_repo", boom)

    main.invalidate_repo_caches("repoA")  # must not raise


def test_invalidate_repo_caches_survives_a_broken_bm25_cache(monkeypatch):
    def boom(repo_id):
        raise RuntimeError("bm25 exploded")

    monkeypatch.setattr(main, "invalidate_bm25_cache", boom)

    main.invalidate_repo_caches("repoA")  # must not raise


def test_invalidate_repo_caches_handles_the_unscoped_bucket(monkeypatch):
    """repo_id=None is a scope like any other, not an error."""
    monkeypatch.setattr(cache.repositories, "get_current_repo_id", lambda: None)
    cache.put("unscoped question", "unscoped answer")

    main.invalidate_repo_caches(None)

    assert cache.get("unscoped question") is None


def test_invalidating_an_unknown_repository_is_a_no_op(scoped):
    scoped("repoA")
    cache.put("a question", "an answer")

    main.invalidate_repo_caches("repoZ")

    assert cache.get("a question") == "an answer"
