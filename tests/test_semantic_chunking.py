"""Tests for semantic source-code chunking introduced by issue #134."""

from pathlib import Path

from indexer import get_file_chunks


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_small_functions_remain_whole_chunks(tmp_path):
    """Neighbouring functions are indexed independently, never split together."""
    path = _write(
        tmp_path,
        "service.py",
        "import os\n\n"
        "def first():\n"
        "    value = 1\n"
        "    return value\n\n"
        "def second():\n"
        "    value = 2\n"
        "    return value\n",
    )

    chunks = get_file_chunks(str(path), chunk_size=6)
    symbols = [chunk for chunk in chunks if chunk.get("is_symbol")]

    first = next(chunk for chunk in symbols if chunk["symbol_name"] == "first")
    second = next(chunk for chunk in symbols if chunk["symbol_name"] == "second")

    assert first["start_line"] == 3
    assert first["end_line"] == 5
    assert "def first" in first["text"]
    assert "def second" not in first["text"]

    assert second["start_line"] == 7
    assert second["end_line"] == 9
    assert "def second" in second["text"]
    assert "def first" not in second["text"]


def test_module_context_is_chunked_separately_from_symbols(tmp_path):
    """Imports/constants stay retrievable without bleeding into a function body."""
    path = _write(
        tmp_path,
        "settings.py",
        "import os\n"
        "TIMEOUT = 30\n\n"
        "def connect():\n"
        "    return TIMEOUT\n",
    )

    chunks = get_file_chunks(str(path), chunk_size=5)
    context_chunks = [
        chunk for chunk in chunks
        if chunk.get("chunk_type") == "module_context"
    ]
    connect = next(
        chunk for chunk in chunks
        if chunk.get("symbol_name") == "connect"
    )

    assert any("TIMEOUT = 30" in chunk["text"] for chunk in context_chunks)
    assert all("def connect" not in chunk["text"] for chunk in context_chunks)
    assert "def connect" in connect["text"]
    assert "TIMEOUT = 30" not in connect["text"]


def test_large_function_splits_only_within_its_boundary(tmp_path):
    """Oversized symbols are partitioned without absorbing adjacent functions."""
    large_body = "".join(f"    value_{i} = {i}\n" for i in range(12))
    path = _write(
        tmp_path,
        "large.py",
        "def large_function():\n"
        + large_body
        + "    return value_11\n\n"
        + "def neighbour():\n"
        + "    return 'safe'\n",
    )

    chunks = get_file_chunks(str(path), chunk_size=6)
    large_parts = [
        chunk for chunk in chunks
        if chunk.get("symbol_name") == "large_function"
    ]
    neighbour = [
        chunk for chunk in chunks
        if chunk.get("symbol_name") == "neighbour"
    ]

    assert len(large_parts) > 1
    assert [chunk["symbol_part"] for chunk in large_parts] == list(
        range(1, len(large_parts) + 1)
    )
    assert all(
        chunk["symbol_parts"] == len(large_parts) for chunk in large_parts
    )
    assert all("def neighbour" not in chunk["text"] for chunk in large_parts)
    assert len(neighbour) == 1
    assert "def neighbour" in neighbour[0]["text"]


def test_symbol_free_python_file_keeps_line_chunk_fallback(tmp_path):
    """Pure constants/comments remain compatible with the old line chunker."""
    path = _write(
        tmp_path,
        "constants.py",
        "# application constants\n"
        "TIMEOUT = 30\n"
        "RETRIES = 3\n",
    )

    chunks = get_file_chunks(str(path), chunk_size=4)

    assert chunks
    assert all(chunk["is_symbol"] is False for chunk in chunks)
    assert "TIMEOUT = 30" in chunks[0]["text"]
