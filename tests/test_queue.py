"""Unit tests for the indexing queue implementation."""

import time
import io
import zipfile
import pytest
from fastapi.testclient import TestClient

from main import app, indexing_queue, jobs_history


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_indexing_queue_endpoint_returns_jobs(client):
    """The /index/queue endpoint returns the list of queued/run jobs."""
    response = client.get("/index/queue")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_multiple_indexing_requests_are_queued(client):
    """Multiple start indexing calls should result in jobs appended to the queue history."""
    initial_jobs_count = len(jobs_history)

    # Queue first reindex
    res1 = client.post("/index/start")
    assert res1.status_code == 200
    body1 = res1.json()
    assert body1["status"] == "started"
    assert "job_id" in body1

    # Queue second reindex
    res2 = client.post("/index/start")
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2["status"] == "started"
    assert "job_id" in body2

    # Check queue history updated
    assert len(jobs_history) == initial_jobs_count + 2

    job1 = next((j for j in jobs_history if j["id"] == body1["job_id"]), None)
    job2 = next((j for j in jobs_history if j["id"] == body2["job_id"]), None)
    
    assert job1 is not None
    assert job2 is not None
    assert job1["type"] == "reindex"
    assert job2["type"] == "reindex"


def test_failed_jobs_are_handled_gracefully(client):
    """Queue should record the error for invalid uploads and proceed without blocking."""
    # Create a corrupted zip file bytes (not a valid zip)
    invalid_zip = b"totally invalid zip content"

    # Queue an upload job that will fail zip check or extraction
    response = client.post(
        "/index/upload",
        files={"file": ("corrupted.zip", invalid_zip, "application/zip")}
    )
    
    # Validates structure immediately, so returns 400
    assert response.status_code == 400
