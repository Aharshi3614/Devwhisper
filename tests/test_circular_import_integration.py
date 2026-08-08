"""Integration tests: the indexer calls the circular import checker.

`index_directory` now parses each indexed file's imports and warns when it
finds a circular import. These tests mock the Qdrant client and embedder
(the same pattern as test_indexer_batching.py) and drive the real indexer,
so we verify the checker is actually wired into the indexing flow.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import indexer


def _configure_indexer(monkeypatch, tmp_path, file_paths):
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = SimpleNamespace(
        tolist=lambda: [0.1, 0.2, 0.3]
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(indexer, "client", mock_client)
    monkeypatch.setattr(indexer, "embedder", mock_embedder)
    monkeypatch.setattr(indexer, "create_collection", MagicMock())
    monkeypatch.setattr(indexer, "load_gitignore_rules", lambda _directory: [])
    monkeypatch.setattr(
        indexer,
        "collect_indexable_files",
        lambda _directory, gitignore_rules=None: (file_paths, []),
    )
    monkeypatch.setattr(
        indexer,
        "get_file_chunks",
        lambda path: [
            {
                "text": f"code from {path}",
                "file": path,
                "start_line": 1,
                "is_symbol": False,
            }
        ],
    )
    monkeypatch.setattr(indexer, "BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
    monkeypatch.setattr(
        indexer,
        "BM25Okapi",
        lambda corpus: {"tokenized_corpus": corpus},
    )

    return mock_client


def _circular_pair(tmp_path):
    """Create a.py and b.py that import each other."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("import b\n", encoding="utf-8")
    b.write_text("import a\n", encoding="utf-8")
    return [str(a), str(b)]


def test_index_directory_warns_on_circular_imports(tmp_path, monkeypatch):
    file_paths = _circular_pair(tmp_path)
    _configure_indexer(monkeypatch, tmp_path, file_paths)

    mock_logger = MagicMock()
    monkeypatch.setattr(indexer, "logger", mock_logger)

    indexer.index_directory(str(tmp_path))

    calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert calls, "expected at least one warning"
    assert any("circular" in c.lower() for c in calls)


def test_index_directory_no_warning_without_circular(tmp_path, monkeypatch):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("import b\n", encoding="utf-8")
    b.write_text("VALUE = 1\n", encoding="utf-8")
    file_paths = [str(a), str(b)]
    _configure_indexer(monkeypatch, tmp_path, file_paths)

    mock_logger = MagicMock()
    monkeypatch.setattr(indexer, "logger", mock_logger)

    indexer.index_directory(str(tmp_path))

    calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert not any("circular" in c.lower() for c in calls)
