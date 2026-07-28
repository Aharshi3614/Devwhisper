"""Unit tests for the codebase indexer, covering Markdown indexing support."""

import os
import tempfile
from types import SimpleNamespace

from config import SUPPORTED_EXTENSIONS
import indexer
from indexer import chunk_file, validate_index


def test_supported_extensions_includes_markdown():
    """.md files must be in the supported extensions set."""
    assert ".md" in SUPPORTED_EXTENSIONS


def test_chunk_file_handles_markdown_content():
    """A Markdown file is chunked line-by-line just like any source file."""
    md_content = """# DevWhisper

A voice-native developer agent.

## Features

- Ask questions about your code
- Get answers in seconds

## Quick Start

Run `pip install -r requirements.txt`.
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        tmp_path = f.name

    try:
        chunks = chunk_file(tmp_path, chunk_size=5)

        assert len(chunks) > 0
        assert chunks[0]["file"] == os.path.basename(tmp_path)
        assert chunks[0]["start_line"] == 1
        assert "# DevWhisper" in chunks[0]["text"]
    finally:
        os.unlink(tmp_path)


def test_chunk_file_skips_empty_markdown():
    """An empty .md file produces no chunks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        tmp_path = f.name

    try:
        chunks = chunk_file(tmp_path)
        assert chunks == []
    finally:
        os.unlink(tmp_path)


def test_validate_index_passes_for_complete_index(monkeypatch, tmp_path):
    """A complete index should validate cleanly."""
    source_file = tmp_path / "sample.py"
    source_file.write_text(
        "\n".join(f"line {i}" for i in range(1, 19)),
        encoding="utf-8",
    )

    chunks = chunk_file(str(source_file))
    file_hash = indexer.get_file_hash(str(source_file))
    points = []

    for chunk_index, chunk in enumerate(chunks):
        payload = indexer._build_chunk_payload(  # pylint: disable=protected-access
            str(source_file),
            chunk,
            chunk_index,
            len(chunks),
            file_hash,
        )
        points.append(
            SimpleNamespace(
                id=indexer._stable_point_id(  # pylint: disable=protected-access
                    str(source_file),
                    chunk["start_line"],
                ),
                payload=payload,
            )
        )

    class FakeClient:
        def collection_exists(self, collection_name):
            return True

        def scroll(self, **kwargs):
            return points, None

    monkeypatch.setattr(indexer, "client", FakeClient())

    report = validate_index(str(tmp_path))

    assert report.is_valid is True
    assert report.expected_chunk_count == len(points)
    assert report.indexed_chunk_count == len(points)
    assert report.missing_point_ids == []
    assert report.unexpected_point_ids == []
    assert report.metadata_issues == []
    assert report.file_issues == []
    assert report.collection_issues == []


def test_validate_index_reports_missing_points_and_bad_metadata(monkeypatch, tmp_path):
    """Validation should flag missing chunks and malformed metadata."""
    source_file = tmp_path / "sample.py"
    source_file.write_text(
        "\n".join(f"line {i}" for i in range(1, 19)),
        encoding="utf-8",
    )

    chunks = chunk_file(str(source_file))
    file_hash = indexer.get_file_hash(str(source_file))
    first_chunk = chunks[0]
    payload = indexer._build_chunk_payload(  # pylint: disable=protected-access
        str(source_file),
        first_chunk,
        0,
        len(chunks),
        file_hash,
    )
    malformed_payload = dict(payload)
    malformed_payload.pop("chunk_hash")

    points = [
        SimpleNamespace(
            id=indexer._stable_point_id(  # pylint: disable=protected-access
                str(source_file),
                first_chunk["start_line"],
            ),
            payload=malformed_payload,
        )
    ]

    class FakeClient:
        def collection_exists(self, collection_name):
            return True

        def scroll(self, **kwargs):
            return points, None

    monkeypatch.setattr(indexer, "client", FakeClient())

    report = validate_index(str(tmp_path))

    assert report.is_valid is False
    assert report.expected_chunk_count == len(chunks)
    assert report.indexed_chunk_count == len(points)
    assert report.missing_point_ids
    assert report.metadata_issues
