"""Unit tests for the codebase ZIP upload endpoint."""

import io
import zipfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from indexer import progress_state


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_progress():
    progress_state.update({
        "running": False,
        "current": 0,
        "total": 0,
        "percent": 0,
        "current_file": "",
        "status": "idle",
        "message": "",
        "skipped": [],
        "skipped_count": 0,
    })
    yield
    progress_state.update({
        "running": False,
        "current": 0,
        "total": 0,
        "percent": 0,
        "current_file": "",
        "status": "idle",
        "message": "",
        "skipped": [],
        "skipped_count": 0,
    })


def test_upload_rejects_non_zip(client):
    """File extension must end with .zip."""
    response = client.post(
        "/index/upload",
        files={"file": ("test.txt", b"some text", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only ZIP archives are supported" in response.json()["message"]


def test_upload_rejects_invalid_zip(client):
    """File content must be a valid ZIP archive."""
    response = client.post(
        "/index/upload",
        files={"file": ("test.zip", b"corrupted content", "application/zip")}
    )
    assert response.status_code == 400
    assert "Invalid ZIP archive" in response.json()["message"]


def test_upload_rejects_path_traversal(client):
    """Path traversal (Zip Slip) vulnerability check."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("../traversal.py", "print('traversal')")
    
    zip_buffer.seek(0)
    response = client.post(
        "/index/upload",
        files={"file": ("traversal.zip", zip_buffer.read(), "application/zip")}
    )
    assert response.status_code == 400
    assert "Path traversal detected in ZIP" in response.json()["message"]


def test_upload_rejects_no_supported_files(client):
    """ZIP must contain at least one supported file extension (.py, .md)."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("readme.txt", "unsupported text file content")
    
    zip_buffer.seek(0)
    response = client.post(
        "/index/upload",
        files={"file": ("unsupported.zip", zip_buffer.read(), "application/zip")}
    )
    assert response.status_code == 400
    assert "No supported files (.py, .md) found in the uploaded ZIP archive" in response.json()["message"]


def test_upload_success_starts_indexing(client):
    """A valid ZIP upload extracts files and triggers background indexing thread."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("app/main.py", "def my_func(): pass")
        zip_file.writestr("docs/readme.md", "# Documentation")

    zip_buffer.seek(0)
    
    with patch("main.threading.Thread") as mock_thread:
        mock_thread.return_value.start.return_value = None
        response = client.post(
            "/index/upload",
            files={"file": ("codebase.zip", zip_buffer.read(), "application/zip")}
        )
        
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "started"
    assert "ZIP file uploaded and extracted successfully" in body["message"]
    mock_thread.assert_called_once()
