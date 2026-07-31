"""Unit tests for the codebase indexer, covering Markdown indexing support."""

import os
import tempfile

from config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
from indexer import chunk_file, collect_indexable_files, get_file_chunks


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
        with open(path, "wb") as f:
            f.write(content.encode())

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

def test_get_file_chunks_includes_symbols_for_python():
    """Python files produce both symbol and line chunks."""
    source = (
        "def preprocess(data):\n"
        '    """Clean data."""\n'
        "    return data.dropna()\n"
        "\n"
        "class Model:\n"
        "    def train(self):\n"
        "        pass\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        tmp_path = f.name

    try:
        chunks = get_file_chunks(tmp_path, chunk_size=5)
        sym_chunks = [c for c in chunks if c.get("is_symbol")]
        line_chunks = [c for c in chunks if not c.get("is_symbol")]

        assert len(sym_chunks) == 3
        names = {c["symbol_name"] for c in sym_chunks}
        assert names == {"preprocess", "Model", "train"}

        assert len(line_chunks) > 0
    finally:
        os.unlink(tmp_path)


def test_get_file_chunks_no_symbols_for_markdown():
    """Markdown files produce only line chunks."""
    md = "# Title\n\nSome text.\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md)
        tmp_path = f.name

    try:
        chunks = get_file_chunks(tmp_path)
        sym_chunks = [c for c in chunks if c.get("is_symbol")]
        line_chunks = [c for c in chunks if not c.get("is_symbol")]

        assert sym_chunks == []
        assert len(line_chunks) > 0
        assert all(c.get("is_symbol") is False for c in line_chunks)
    finally:
        os.unlink(tmp_path)


def test_symbol_chunk_has_expected_metadata():
    """Symbol chunks carry the metadata fields the retriever needs."""
    source = (
        'class Processor:\n'
        '    """Process things."""\n'
        "\n"
        "    def run(self):\n"
        '        """Run it."""\n'
        "        pass\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        tmp_path = f.name

    try:
        chunks = get_file_chunks(tmp_path)
        sym = next(c for c in chunks if c.get("symbol_name") == "run")
        assert sym["symbol_type"] == "method"
        assert sym["parent_class"] == "Processor"
        assert sym["docstring"] == "Run it."
        assert sym["start_line"] == 4
        assert sym["end_line"] == 6
        assert sym["is_symbol"] is True
    finally:
        os.unlink(tmp_path)
