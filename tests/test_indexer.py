"""Unit tests for the codebase indexer, covering Markdown indexing support."""

import os
import tempfile

from config import SUPPORTED_EXTENSIONS
from indexer import chunk_file


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