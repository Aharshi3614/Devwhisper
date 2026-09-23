"""Tests for automatic code language detection (Issue #137)."""

import os
import tempfile

from language_detector import detect_language, get_language_spec, is_code_file
from indexer import chunk_file


# ---------------------------------------------------------------------------
# detect_language tests
# ---------------------------------------------------------------------------

def test_detect_python():
    assert detect_language("main.py") == "python"
    assert detect_language("utils.pyw") == "python"
    assert detect_language("types.pyi") == "python"


def test_detect_javascript():
    assert detect_language("app.js") == "javascript"
    assert detect_language("App.jsx") == "javascript"
    assert detect_language("worker.mjs") == "javascript"
    assert detect_language("config.cjs") == "javascript"


def test_detect_typescript():
    assert detect_language("index.ts") == "typescript"
    assert detect_language("Component.tsx") == "typescript"
    assert detect_language("util.mts") == "typescript"
    assert detect_language("types.cts") == "typescript"


def test_detect_go():
    assert detect_language("main.go") == "go"


def test_detect_rust():
    assert detect_language("lib.rs") == "rust"
    assert detect_language("main.rs") == "rust"


def test_detect_java():
    assert detect_language("App.java") == "java"


def test_detect_markdown():
    assert detect_language("README.md") == "markdown"
    assert detect_language("docs.markdown") == "markdown"


def test_detect_unknown_extension():
    assert detect_language("image.png") is None
    assert detect_language("data.csv") is None
    assert detect_language("noextension") is None


# ---------------------------------------------------------------------------
# get_language_spec tests
# ---------------------------------------------------------------------------

def test_get_language_spec_python():
    spec = get_language_spec("python")
    assert spec is not None
    assert spec.name == "python"
    assert ".py" in spec.extensions
    assert "#" in spec.line_comment_prefixes
    assert spec.is_code is True


def test_get_language_spec_markdown():
    spec = get_language_spec("markdown")
    assert spec is not None
    assert spec.name == "markdown"
    assert spec.is_code is False


def test_get_language_spec_unknown():
    assert get_language_spec("brainfuck") is None


# ---------------------------------------------------------------------------
# is_code_file tests
# ---------------------------------------------------------------------------

def test_is_code_file_python():
    assert is_code_file("main.py") is True


def test_is_code_file_markdown():
    assert is_code_file("README.md") is False


def test_is_code_file_unknown():
    assert is_code_file("image.png") is False


# ---------------------------------------------------------------------------
# Chunk language metadata tests
# ---------------------------------------------------------------------------

def test_chunk_file_includes_language():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def hello():\n    pass\n")
        tmp_path = f.name

    try:
        chunks = chunk_file(tmp_path, chunk_size=5)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "language" in chunk
            assert chunk["language"] == "python"
    finally:
        os.unlink(tmp_path)


def test_chunk_file_includes_language_js():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write("function hello() {\n  return true;\n}\n")
        tmp_path = f.name

    try:
        chunks = chunk_file(tmp_path, chunk_size=5)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["language"] == "javascript"
    finally:
        os.unlink(tmp_path)


def test_chunk_file_includes_language_go():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
        f.write("package main\n\nfunc main() {\n}\n")
        tmp_path = f.name

    try:
        chunks = chunk_file(tmp_path, chunk_size=5)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["language"] == "go"
    finally:
        os.unlink(tmp_path)


def test_chunk_file_includes_language_markdown():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Title\n\nSome content here.\n")
        tmp_path = f.name

    try:
        chunks = chunk_file(tmp_path, chunk_size=5)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["language"] == "markdown"
    finally:
        os.unlink(tmp_path)
