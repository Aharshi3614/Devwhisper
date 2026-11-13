"""tests/test_indexing_job_manager.py — Unit and integration tests for indexing job manager."""

import queue
import threading
import time
import pytest
from fastapi.testclient import TestClient

from indexing_job_manager import IndexingJobManager, JobStatus, IndexingJob
from pipeline_validator import IndexingPipelineTracker, validate_indexing_sequence, PipelineStageError


def test_indexing_job_lifecycle():
    manager = IndexingJobManager(max_history=10)
    job = manager.create_job(repo_id="repo_alpha", is_dry_run=True)

    assert job.status == JobStatus.PENDING
    assert job.repo_id == "repo_alpha"
    assert job.is_dry_run is True

    # Test state transitions
    job.status = JobStatus.RUNNING
    job.percent = 50
    assert job.to_dict()["status"] == "running"
    assert job.to_dict()["percent"] == 50

    # Cancellation
    assert job.cancel() is True
    assert job.status == JobStatus.CANCELLED
    assert job.is_cancelled() is True

    # Second cancel is no-op
    assert job.cancel() is False


def test_indexing_job_manager_active_lookup():
    manager = IndexingJobManager()
    job1 = manager.create_job(repo_id="repo_1")
    job1.status = JobStatus.RUNNING

    active = manager.get_active_job_for_repo("repo_1")
    assert active is not None
    assert active.id == job1.id

    job1.status = JobStatus.COMPLETED
    assert manager.get_active_job_for_repo("repo_1") is None


def test_indexing_pipeline_validator():
    # Valid forward sequence
    validate_indexing_sequence(["collect_files", "extract_symbols", "generate_chunks", "persist_cache"])

    # Invalid backwards transition
    tracker = IndexingPipelineTracker()
    tracker.enter("collect_files")
    tracker.enter("extract_symbols")
    with pytest.raises(PipelineStageError):
        tracker.enter("collect_files")



def test_indexing_jobs_api_endpoints(monkeypatch):
    from main import app

    client = TestClient(app)

    # Start job
    res_start = client.post("/index/jobs/start", json={"repo_id": "test_repo", "dry_run": True})
    assert res_start.status_code == 200
    job_data = res_start.json()["job"]
    job_id = job_data["id"]

    # List jobs
    res_list = client.get("/index/jobs")
    assert res_list.status_code == 200
    jobs = res_list.json()["jobs"]
    assert any(j["id"] == job_id for j in jobs)

    # Get status
    res_status = client.get(f"/index/jobs/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["job"]["id"] == job_id

    # Cancel job
    res_cancel = client.post(f"/index/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["job"]["status"] in ("cancelled", "completed", "failed")

    # 404 for unknown job
    res_404 = client.get("/index/jobs/nonexistent-id")
    assert res_404.status_code == 404


def _run_worker_until_stopped(queue_obj):
    """Run queue_worker for the jobs already queued, then stop it."""
    import main

    worker = threading.Thread(target=main.queue_worker, daemon=True)
    worker.start()
    queue_obj.put(None)
    worker.join(timeout=5)
    assert not worker.is_alive(), "queue_worker did not stop"


def test_worker_skips_job_cancelled_before_start(monkeypatch):
    """A job cancelled while queued must not run or stay active."""
    import main
    from indexing_job_manager import job_manager

    work_queue = queue.Queue()
    indexed = []
    monkeypatch.setattr(main, "indexing_queue", work_queue)
    monkeypatch.setattr(main, "index_directory", lambda *a, **k: indexed.append(1))
    monkeypatch.setattr(main, "invalidate_repo_caches", lambda *a, **k: None)

    job = job_manager.create_job(repo_id="cancelled-before-start")
    assert job_manager.cancel_job(job.id) is True

    record = {
        "id": job.id,
        "type": "reindex",
        "name": "Cancelled before start",
        "repo_id": "cancelled-before-start",
        "dry_run": False,
        "job_instance": job,
    }
    work_queue.put(record)
    _run_worker_until_stopped(work_queue)

    assert indexed == [], "cancelled job must not start index_directory()"
    assert record["status"] == "cancelled"
    assert job.status == JobStatus.CANCELLED
    assert job_manager.get_active_job_for_repo("cancelled-before-start") is None


def test_worker_releases_job_cancelled_during_run(monkeypatch):
    """A cancellation that arrives mid-run must be written back to the job."""
    import main
    from indexing_job_manager import job_manager

    work_queue = queue.Queue()
    monkeypatch.setattr(main, "indexing_queue", work_queue)
    monkeypatch.setattr(main, "invalidate_repo_caches", lambda *a, **k: None)

    job = job_manager.create_job(repo_id="cancelled-during-run")
    record = {
        "id": job.id,
        "type": "reindex",
        "name": "Cancelled during run",
        "repo_id": "cancelled-during-run",
        "dry_run": False,
        "job_instance": job,
    }

    def cancel_during_index(*args, **kwargs):
        # Simulates index_directory() observing the cancellation event and
        # returning early after the user has cancelled the job.
        job_manager.cancel_job(job.id)
        return None

    monkeypatch.setattr(main, "index_directory", cancel_during_index)

    work_queue.put(record)
    _run_worker_until_stopped(work_queue)

    assert record["status"] == "cancelled"
    assert job.status == JobStatus.CANCELLED
    assert job_manager.get_active_job_for_repo("cancelled-during-run") is None
