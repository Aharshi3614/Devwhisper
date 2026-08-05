"""Unit tests for the codebase indexer, covering Markdown indexing support."""

import os
import tempfile

from config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
from indexer import (
    chunk_file,
    collect_indexable_files,
    get_file_chunks,
    load_gitignore_rules,
)


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
    """Unsupported-extension files are excluded from indexing AND reported.

    Issue #223: previously these files were silently dropped. They are now
    recorded in the ``skipped`` list with reason ``unsupported_extension``
    so operators can see exactly what was excluded and why.
    """
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
        # The .txt file must now appear in the skipped list with the
        # correct reason and a human-readable detail field.
        assert len(skipped) == 1
        assert skipped[0]["path"] == txt_path
        assert skipped[0]["reason"] == "unsupported_extension"
        assert skipped[0]["detail"] == ".txt"


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


# --- .gitignore awareness ---


def test_load_gitignore_rules_empty_when_no_gitignore():
    """A tree without any .gitignore produces no rules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write("x = 1\n")
        assert load_gitignore_rules(tmpdir) == []


def test_collect_respects_gitignore():
    """Files/directories listed in .gitignore are excluded from indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ignored_dir = os.path.join(tmpdir, "ignored")
        os.makedirs(ignored_dir)
        with open(os.path.join(tmpdir, "keep.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(ignored_dir, "secret.py"), "w") as f:
            f.write("secret = 1\n")
        with open(os.path.join(tmpdir, ".gitignore"), "w") as f:
            f.write("ignored/\nsecret.py\n")

        rules = load_gitignore_rules(tmpdir)
        files, skipped = collect_indexable_files(tmpdir, gitignore_rules=rules)

        assert os.path.join(tmpdir, "keep.py") in files
        assert os.path.join(ignored_dir, "secret.py") not in files
        assert any(s["reason"] == "gitignored" for s in skipped)


def test_collect_respects_nested_gitignore():
    """A nested .gitignore only affects files under its own directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(subdir, "keep.py"), "w") as f:
            f.write("y = 2\n")
        with open(os.path.join(subdir, "drop.py"), "w") as f:
            f.write("z = 3\n")
        with open(os.path.join(subdir, ".gitignore"), "w") as f:
            f.write("drop.py\n")

        rules = load_gitignore_rules(tmpdir)
        files, _ = collect_indexable_files(tmpdir, gitignore_rules=rules)

        assert os.path.join(tmpdir, "main.py") in files
        assert os.path.join(subdir, "keep.py") in files
        assert os.path.join(subdir, "drop.py") not in files


def test_gitignore_negation_honored():
    """`!pattern` inside a .gitignore re-includes a previously ignored file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "keep.py"), "w") as f:
            f.write("keep = 1\n")
        with open(os.path.join(tmpdir, "other.py"), "w") as f:
            f.write("other = 1\n")
        with open(os.path.join(tmpdir, ".gitignore"), "w") as f:
            f.write("*.py\n!keep.py\n")

        rules = load_gitignore_rules(tmpdir)
        files, _ = collect_indexable_files(tmpdir, gitignore_rules=rules)

        assert os.path.join(tmpdir, "keep.py") in files
        assert os.path.join(tmpdir, "other.py") not in files


def test_collect_unaffected_without_gitignore_rules():
    """Passing no rules leaves the existing collection behavior unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "ignored_dir"), exist_ok=True)
        with open(os.path.join(tmpdir, "good.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(tmpdir, "ignored_dir", "inner.py"), "w") as f:
            f.write("y = 1\n")

        files, skipped = collect_indexable_files(tmpdir, max_bytes=1000)

        assert len(files) == 2
        assert skipped == []
