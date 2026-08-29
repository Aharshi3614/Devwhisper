"""tests/test_indexing_job_manager.py — Unit and integration tests for indexing job manager."""

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
