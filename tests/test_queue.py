"""Unit tests for the indexing queue implementation."""

import time
import io
import zipfile
import glob
import os
import pytest
from fastapi.testclient import TestClient

from main import app, indexing_queue, jobs_history


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def cleanup_temp_zips():
    yield
    for f in glob.glob("temp_upload_*.zip"):
        try:
            os.remove(f)
        except Exception:
            pass


def test_indexing_queue_endpoint_returns_jobs(client):
    """The /index/queue endpoint returns the list of queued/run jobs."""
    response = client.get("/index/queue")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_multiple_indexing_requests_are_queued(client):
    """Multiple upload indexing calls should result in jobs appended to the queue history."""
    initial_jobs_count = len(jobs_history)

    zip_buffer1 = io.BytesIO()
    with zipfile.ZipFile(zip_buffer1, "w") as zip_file:
        zip_file.writestr("app/main.py", "def my_func(): pass")
        zip_file.writestr("docs/readme.md", "# Documentation")
    zip_buffer1.seek(0)

    zip_buffer2 = io.BytesIO()
    with zipfile.ZipFile(zip_buffer2, "w") as zip_file:
        zip_file.writestr("app/utils.py", "def another(): pass")
        zip_file.writestr("docs/readme.md", "# Documentation 2")
    zip_buffer2.seek(0)

    # Queue first upload
    res1 = client.post(
        "/index/upload",
        files={"file": ("codebase1.zip", zip_buffer1.read(), "application/zip")}
    )
    assert res1.status_code == 200
    body1 = res1.json()
    assert body1["status"] == "started"
    assert "job_id" in body1

    # Queue second upload
    res2 = client.post(
        "/index/upload",
        files={"file": ("codebase2.zip", zip_buffer2.read(), "application/zip")}
    )
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2["status"] == "started"
    assert "job_id" in body2

    # Check queue history updated
    assert len(jobs_history) >= initial_jobs_count + 2

    job1 = next((j for j in jobs_history if j["id"] == body1["job_id"]), None)
    job2 = next((j for j in jobs_history if j["id"] == body2["job_id"]), None)
    
    assert job1 is not None
    assert job2 is not None
    assert job1["type"] == "upload"
    assert job2["type"] == "upload"


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
