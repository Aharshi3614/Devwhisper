"""Unit tests for the codebase indexer, covering Markdown indexing support."""

import os
import tempfile

from config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
from indexer import chunk_file, collect_indexable_files


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


# --- collect_indexable_files ---

def test_collect_skips_oversized_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        small_path = os.path.join(tmpdir, "small.py")
        large_path = os.path.join(tmpdir, "large.py")
        with open(small_path, "w") as f:
            f.write("x = 1\n")
        with open(large_path, "w") as f:
            f.write("x\n" * 200)

        files, skipped = collect_indexable_files(tmpdir, max_bytes=100)

        assert small_path in files
        assert large_path not in files
        assert len(skipped) == 1
        assert skipped[0]["path"] == large_path
        assert skipped[0]["reason"] == "oversized"


def test_collect_keeps_file_at_exact_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "exact.py")
        content = "x\n" * 50
        with open(path, "w") as f:
            f.write(content)

        files, skipped = collect_indexable_files(tmpdir, max_bytes=len(content))

        assert path in files
        assert len(skipped) == 0


def test_collect_filters_unsupported_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = os.path.join(tmpdir, "good.py")
        txt_path = os.path.join(tmpdir, "bad.txt")
        with open(py_path, "w") as f:
            f.write("x = 1\n")
        with open(txt_path, "w") as f:
            f.write("hello\n")

        files, skipped = collect_indexable_files(tmpdir, max_bytes=1000)

        assert py_path in files
        assert txt_path not in files
        assert len(skipped) == 0


def test_collect_config_default():
    assert MAX_FILE_SIZE_BYTES == MAX_FILE_SIZE_MB * 1024 * 1024
    assert MAX_FILE_SIZE_MB == 1