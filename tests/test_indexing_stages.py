"""Tests for decoupled indexing stages (Issue #191)."""
import pytest
from unittest.mock import MagicMock, patch
import indexer

def test_decoupled_indexing_stages_exist():
    assert hasattr(indexer, "discover_files")
    assert hasattr(indexer, "generate_chunks")
    assert hasattr(indexer, "generate_embeddings")
    assert hasattr(indexer, "upload_vectors")

def test_discover_files(tmp_path):
    f1 = tmp_path / "a.py"
    f1.write_text("print('hello')")
    files, skipped = indexer.discover_files(str(tmp_path))
    assert len(files) == 1
    assert files[0] == str(f1)

def test_generate_chunks(tmp_path):
    f1 = tmp_path / "a.py"
    f1.write_text("def hello(): pass")
    chunks, cache_map, imports = indexer.generate_chunks([str(f1)])
    assert len(chunks) > 0
    assert str(f1) in cache_map

@patch("indexer.embedder")
def test_generate_embeddings(mock_embedder):
    mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384
    chunks = [{"text": "code", "file": "a.py", "start_line": 1}]
    points = indexer.generate_embeddings(chunks, "test_repo")
    assert len(points) == 1
    assert points[0].payload["repository"] == "test_repo"

@patch("indexer.client")
def test_upload_vectors(mock_client):
    mock_point = MagicMock()
    uploaded = indexer.upload_vectors("test_coll", [mock_point])
    assert uploaded == 1
    mock_client.upsert.assert_called_once()
