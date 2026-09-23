"""Tests for post-index validation (Issue #139)."""

import json
import os
import pickle
import tempfile

from index_validator import (
    validate_index,
    validate_metadata_integrity,
    validate_cache_schema,
    validate_chunk_counts,
)


def _make_valid_cache(tmpdir: str) -> tuple[str, dict]:
    """Create a valid cache file and return its path and data."""
    cache_data = {
        "_metadata": {
            "repository_name": "test_repo",
            "indexing_timestamp": "2026-01-01T00:00:00",
            "indexing_duration_seconds": 1.5,
            "indexed_file_count": 1,
            "skipped_file_count": 0,
            "skipped_files": [],
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_version": "v2",
            "max_file_size_mb": 1,
            "chunk_statistics": {
                "total_chunks": 5,
                "average_size": 10.0,
                "largest": None,
                "smallest": None,
            },
        },
        "/path/to/file.py": {
            "mtime": 1700000000.0,
            "hash": "abc123",
            "symbols": [],
        },
    }
    cache_path = os.path.join(tmpdir, "cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    return cache_path, cache_data


def _make_valid_bm25(tmpdir: str, chunk_count: int = 5) -> str:
    """Create a valid BM25 pickle file and return its path."""
    bm25_data = {
        "bm25": None,
        "corpus": ["chunk text"] * chunk_count,
        "chunks": [{"text": "chunk text", "file": "file.py"}] * chunk_count,
    }
    bm25_path = os.path.join(tmpdir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    return bm25_path


# ---------------------------------------------------------------------------
# validate_metadata_integrity tests
# ---------------------------------------------------------------------------

def test_valid_metadata():
    cache_data = {
        "_metadata": {
            "repository_name": "test",
            "indexing_timestamp": "2026-01-01T00:00:00",
            "indexing_duration_seconds": 1.0,
            "indexed_file_count": 0,
            "skipped_file_count": 0,
            "skipped_files": [],
            "embedding_model": "model",
            "embedding_version": "v1",
            "max_file_size_mb": 1,
            "chunk_statistics": {},
        },
    }
    errors = validate_metadata_integrity(cache_data)
    assert errors == []


def test_missing_metadata_block():
    errors = validate_metadata_integrity({})
    assert len(errors) == 1
    assert errors[0]["check"] == "metadata_exists"


def test_missing_metadata_field():
    cache_data = {
        "_metadata": {
            "repository_name": "test",
            "indexing_timestamp": "2026-01-01T00:00:00",
            "indexing_duration_seconds": 1.0,
            "indexed_file_count": 0,
            "skipped_file_count": 0,
            "skipped_files": [],
            "embedding_model": "model",
            "embedding_version": "v1",
            # missing max_file_size_mb and chunk_statistics
        },
    }
    errors = validate_metadata_integrity(cache_data)
    assert len(errors) == 2
    field_names = [e["check"] for e in errors]
    assert "metadata_field" in field_names


def test_wrong_metadata_type():
    cache_data = {
        "_metadata": {
            "repository_name": 123,  # should be str
            "indexing_timestamp": "2026-01-01T00:00:00",
            "indexing_duration_seconds": 1.0,
            "indexed_file_count": 0,
            "skipped_file_count": 0,
            "skipped_files": [],
            "embedding_model": "model",
            "embedding_version": "v1",
            "max_file_size_mb": 1,
            "chunk_statistics": {},
        },
    }
    errors = validate_metadata_integrity(cache_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "metadata_type"


# ---------------------------------------------------------------------------
# validate_cache_schema tests
# ---------------------------------------------------------------------------

def test_valid_cache_entries():
    cache_data = {
        "_metadata": {},
        "file.py": {"mtime": 1.0, "hash": "abc", "symbols": []},
        "other.py": {"mtime": 2.0, "hash": "def", "symbols": []},
    }
    errors = validate_cache_schema(cache_data)
    assert errors == []


def test_missing_required_field():
    cache_data = {
        "_metadata": {},
        "file.py": {"mtime": 1.0},  # missing hash
    }
    errors = validate_cache_schema(cache_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "cache_entry_field"


def test_wrong_field_type():
    cache_data = {
        "_metadata": {},
        "file.py": {"mtime": "not_a_float", "hash": "abc"},
    }
    errors = validate_cache_schema(cache_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "cache_entry_type"


def test_non_dict_entry():
    cache_data = {
        "_metadata": {},
        "file.py": "not_a_dict",
    }
    errors = validate_cache_schema(cache_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "cache_entry_type"


# ---------------------------------------------------------------------------
# validate_chunk_counts tests
# ---------------------------------------------------------------------------

def test_matching_counts():
    cache_data = {
        "_metadata": {
            "indexed_file_count": 2,
            "chunk_statistics": {"total_chunks": 10},
        },
        "f1.py": {},
        "f2.py": {},
    }
    bm25_data = {"chunks": [{}] * 10}
    errors = validate_chunk_counts(cache_data, bm25_data)
    assert errors == []


def test_file_count_mismatch():
    cache_data = {
        "_metadata": {
            "indexed_file_count": 5,
            "chunk_statistics": {"total_chunks": 10},
        },
        "f1.py": {},
        "f2.py": {},
    }
    bm25_data = {"chunks": [{}] * 10}
    errors = validate_chunk_counts(cache_data, bm25_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "file_count_mismatch"


def test_chunk_count_mismatch():
    cache_data = {
        "_metadata": {
            "indexed_file_count": 1,
            "chunk_statistics": {"total_chunks": 20},
        },
        "f1.py": {},
    }
    bm25_data = {"chunks": [{}] * 10}
    errors = validate_chunk_counts(cache_data, bm25_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "chunk_count_mismatch"


def test_empty_bm25_with_files():
    cache_data = {
        "_metadata": {
            "indexed_file_count": 2,
            "chunk_statistics": {"total_chunks": 0},
        },
        "f1.py": {},
        "f2.py": {},
    }
    bm25_data = {"chunks": []}
    errors = validate_chunk_counts(cache_data, bm25_data)
    assert len(errors) == 1
    assert errors[0]["check"] == "empty_bm25"


# ---------------------------------------------------------------------------
# validate_index end-to-end tests
# ---------------------------------------------------------------------------

def test_validate_index_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path, _ = _make_valid_cache(tmpdir)
        bm25_path = _make_valid_bm25(tmpdir)

        report = validate_index(cache_path, bm25_path)
        assert report["valid"] is True
        assert report["errors"] == []
        assert report["summary"]["indexed_file_count"] == 1
        assert report["summary"]["bm25_chunk_count"] == 5


def test_validate_index_missing_cache():
    report = validate_index("/nonexistent/cache.json", "/nonexistent/bm25.pkl")
    assert report["valid"] is False
    assert len(report["errors"]) == 1
    assert report["errors"][0]["check"] == "cache_load"


def test_validate_index_missing_bm25():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path, _ = _make_valid_cache(tmpdir)
        report = validate_index(cache_path, "/nonexistent/bm25.pkl")
        assert report["valid"] is False
        assert len(report["errors"]) == 1
        assert report["errors"][0]["check"] == "bm25_load"


def test_validate_index_corrupt_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.json")
        with open(cache_path, "w") as f:
            f.write("not valid json {{{")

        bm25_path = _make_valid_bm25(tmpdir)
        report = validate_index(cache_path, bm25_path)
        assert report["valid"] is False


def test_validate_index_metadata_issues():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_data = {
            "_metadata": {
                "repository_name": "test",
                # missing other required fields
            },
            "file.py": {"mtime": 1.0, "hash": "abc"},
        }
        cache_path = os.path.join(tmpdir, "cache.json")
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        bm25_path = _make_valid_bm25(tmpdir, chunk_count=1)
        report = validate_index(cache_path, bm25_path)
        assert report["valid"] is False
        assert len(report["errors"]) > 0
