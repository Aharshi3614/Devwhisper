"""Tests for bounded Qdrant uploads during codebase indexing."""

import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import indexer


def _create_python_files(directory, count):
    paths = []
    for number in range(count):
        path = directory / f"module_{number:02d}.py"
        path.write_text(f"VALUE = {number}\n", encoding="utf-8")
        paths.append(str(path))
    return paths


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
        lambda path, **_kwargs: [
            {
                "text": f"code from {path}",
                "file": path,
                "path": path,
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


def test_index_directory_uploads_points_in_twenty_file_batches(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Forty-five files are uploaded as batches of 20, 20, and 5."""
    file_paths = _create_python_files(tmp_path, 45)
    mock_client = _configure_indexer(monkeypatch, tmp_path, file_paths)

    monkeypatch.setattr(indexer, "INDEX_FILE_BATCH_SIZE", 20)

    indexer.index_directory(str(tmp_path))

    assert mock_client.upsert.call_count == 3
    assert [
        len(call.kwargs["points"])
        for call in mock_client.upsert.call_args_list
    ] == [20, 20, 5]

    output = capsys.readouterr().out
    assert "Indexed 20/45 files" in output
    assert "Indexed 40/45 files" in output
    assert "Indexed 45/45 files" in output
    assert "Done. Indexed 45 total chunks into Qdrant." in output


def test_index_directory_preserves_small_codebase_behavior(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A repository smaller than one batch still uses one Qdrant upsert."""
    file_paths = _create_python_files(tmp_path, 3)
    mock_client = _configure_indexer(monkeypatch, tmp_path, file_paths)

    monkeypatch.setattr(indexer, "INDEX_FILE_BATCH_SIZE", 20)

    indexer.index_directory(str(tmp_path))

    mock_client.upsert.assert_called_once()
    assert len(mock_client.upsert.call_args.kwargs["points"]) == 3
    assert "Indexed 3/3 files" in capsys.readouterr().out


def test_index_directory_stamps_repository_tag(tmp_path, monkeypatch):
    """Every chunk stored in Qdrant and BM25 carries the repository tag.

    This tag is what makes the shared-index repository filtering (PR #212)
    actually usable — without it, ``repository`` filters match nothing.
    """
    file_paths = _create_python_files(tmp_path, 2)
    mock_client = _configure_indexer(monkeypatch, tmp_path, file_paths)

    indexer.index_directory(str(tmp_path))

    # Qdrant payloads are stamped with the repository's basename.
    upserted = []
    for call in mock_client.upsert.call_args_list:
        upserted.extend(call.kwargs["points"])
    assert upserted, "expected at least one upserted point"
    for point in upserted:
        assert point.payload.get("repository") == tmp_path.name

    # BM25 chunks are stamped as well.
    with open(indexer.BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    assert bm25_data["chunks"], "expected BM25 chunks to be written"
    for chunk in bm25_data["chunks"]:
        assert chunk.get("repository") == tmp_path.name
