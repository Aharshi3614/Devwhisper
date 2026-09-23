"""index_validator.py — Post-index validation for DevWhisper.

This module validates the integrity of indexes after indexing completes,
checking for missing, corrupted, or incomplete data across Qdrant, BM25,
and the index cache.

Key functions:
  - validate_index(): Run all validations and return a diagnostics report.
  - validate_chunk_counts(): Check Qdrant/BM25/cache chunk count consistency.
  - validate_metadata_integrity(): Verify _metadata block completeness.
  - validate_cache_schema(): Verify per-file cache entries have required keys.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

from logger import logger

# Required fields in _metadata block
REQUIRED_METADATA_FIELDS = {
    "repository_name": str,
    "indexing_timestamp": str,
    "indexing_duration_seconds": (int, float),
    "indexed_file_count": int,
    "skipped_file_count": int,
    "skipped_files": list,
    "embedding_model": str,
    "embedding_version": str,
    "max_file_size_mb": (int, float),
    "chunk_statistics": dict,
}

# Required fields in per-file cache entries
REQUIRED_FILE_FIELDS = {
    "mtime": (int, float),
    "hash": str,
}


def validate_index(
    cache_path: str,
    bm25_path: str,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Run all post-index validations and return a diagnostics report.

    Args:
        cache_path: Path to the .index_cache.json file.
        bm25_path: Path to the BM25 pickle file.
        collection_name: Qdrant collection name (optional, skips Qdrant checks if None).

    Returns:
        A dict with keys:
            - "valid": bool — True if no errors found
            - "errors": list[dict] — Critical issues found
            - "warnings": list[str] — Non-critical issues found
            - "summary": dict — Counts of files, chunks, etc.
    """
    report: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    # Load cache data
    cache_data = _load_json(cache_path)
    if cache_data is None:
        report["valid"] = False
        report["errors"].append({
            "check": "cache_load",
            "message": f"Failed to load index cache from {cache_path}",
        })
        return report

    # Load BM25 data
    bm25_data = _load_pickle(bm25_path)
    if bm25_data is None:
        report["valid"] = False
        report["errors"].append({
            "check": "bm25_load",
            "message": f"Failed to load BM25 index from {bm25_path}",
        })
        return report

    # Run validations
    metadata_errors = validate_metadata_integrity(cache_data)
    file_errors = validate_cache_schema(cache_data)
    count_errors = validate_chunk_counts(cache_data, bm25_data)

    report["errors"].extend(metadata_errors)
    report["errors"].extend(file_errors)
    report["errors"].extend(count_errors)

    # Build summary
    metadata = cache_data.get("_metadata", {})
    file_count = sum(1 for k in cache_data if k != "_metadata")
    bm25_chunk_count = len(bm25_data.get("chunks", []))
    cache_chunk_stats = metadata.get("chunk_statistics", {})
    reported_chunk_count = cache_chunk_stats.get("total_chunks", 0)

    report["summary"] = {
        "indexed_file_count": file_count,
        "reported_file_count": metadata.get("indexed_file_count"),
        "bm25_chunk_count": bm25_chunk_count,
        "reported_chunk_count": reported_chunk_count,
        "skipped_file_count": metadata.get("skipped_file_count", 0),
    }

    if report["errors"]:
        report["valid"] = False

    if report["valid"]:
        logger.info("Index validation passed: %d files, %d chunks", file_count, bm25_chunk_count)
    else:
        logger.warning(
            "Index validation found %d issue(s): %s",
            len(report["errors"]),
            [e["message"] for e in report["errors"]],
        )

    return report


def validate_metadata_integrity(cache_data: dict) -> list[dict]:
    """Verify the _metadata block has all required fields with correct types.

    Args:
        cache_data: The full cache data dict (including _metadata).

    Returns:
        List of error dicts. Empty if valid.
    """
    errors = []
    metadata = cache_data.get("_metadata")

    if metadata is None:
        errors.append({
            "check": "metadata_exists",
            "message": "Missing _metadata block in index cache",
        })
        return errors

    for field, expected_type in REQUIRED_METADATA_FIELDS.items():
        if field not in metadata:
            errors.append({
                "check": "metadata_field",
                "message": f"Missing required metadata field: {field}",
            })
        elif not isinstance(metadata[field], expected_type):
            errors.append({
                "check": "metadata_type",
                "message": (
                    f"Metadata field '{field}' has wrong type: "
                    f"expected {expected_type}, got {type(metadata[field]).__name__}"
                ),
            })

    return errors


def validate_cache_schema(cache_data: dict) -> list[dict]:
    """Verify each per-file cache entry has required keys.

    Args:
        cache_data: The full cache data dict.

    Returns:
        List of error dicts. Empty if valid.
    """
    errors = []

    for key, value in cache_data.items():
        if key == "_metadata":
            continue

        if not isinstance(value, dict):
            errors.append({
                "check": "cache_entry_type",
                "message": f"Cache entry for '{key}' is not a dict",
            })
            continue

        for field, expected_type in REQUIRED_FILE_FIELDS.items():
            if field not in value:
                errors.append({
                    "check": "cache_entry_field",
                    "message": f"Cache entry for '{key}' missing field: {field}",
                })
            elif not isinstance(value[field], expected_type):
                errors.append({
                    "check": "cache_entry_type",
                    "message": (
                        f"Cache entry '{key}' field '{field}' has wrong type: "
                        f"expected {expected_type}, got {type(value[field]).__name__}"
                    ),
                })

    return errors


def validate_chunk_counts(
    cache_data: dict,
    bm25_data: dict,
) -> list[dict]:
    """Check that chunk counts are consistent across cache and BM25.

    Args:
        cache_data: The full cache data dict.
        bm25_data: The loaded BM25 pickle data.

    Returns:
        List of error dicts. Empty if consistent.
    """
    errors = []
    metadata = cache_data.get("_metadata", {})
    cache_file_count = sum(1 for k in cache_data if k != "_metadata")
    reported_file_count = metadata.get("indexed_file_count")

    if reported_file_count is not None and cache_file_count != reported_file_count:
        errors.append({
            "check": "file_count_mismatch",
            "message": (
                f"File count mismatch: cache has {cache_file_count} files, "
                f"but metadata reports {reported_file_count}"
            ),
        })

    bm25_chunks = bm25_data.get("chunks", [])
    bm25_chunk_count = len(bm25_chunks)
    reported_chunk_count = metadata.get("chunk_statistics", {}).get("total_chunks")

    if reported_chunk_count is not None and bm25_chunk_count != reported_chunk_count:
        errors.append({
            "check": "chunk_count_mismatch",
            "message": (
                f"Chunk count mismatch: BM25 has {bm25_chunk_count} chunks, "
                f"but metadata reports {reported_chunk_count}"
            ),
        })

    if bm25_chunk_count == 0 and cache_file_count > 0:
        errors.append({
            "check": "empty_bm25",
            "message": "BM25 index has 0 chunks but cache has indexed files",
        })

    return errors


def _load_json(path: str) -> dict | None:
    """Load a JSON file, returning None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_pickle(path: str) -> dict | None:
    """Load a pickle file, returning None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, OSError, EOFError):
        return None
