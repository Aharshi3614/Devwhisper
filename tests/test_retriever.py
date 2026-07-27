"""Unit tests for the Qdrant-backed code retrieval formatter."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import retriever


def _point(payload):
    """Build a lightweight object matching Qdrant's returned point shape."""
    return SimpleNamespace(payload=payload)


def test_retrieve_formats_ranked_results(monkeypatch):
    """Relevant Qdrant points are returned as readable, ranked context."""
    vector = [0.1, 0.2, 0.3]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = vector

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point(
            {
                "file": "pipeline.py",
                "start_line": 12,
                "text": "def preprocess(data):\n    return data.dropna()",
            }
        ),
        _point(
            {
                "file": "model.py",
                "start_line": 31,
                "text": "def train_model(features):\n    return features",
            }
        ),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("How is the data prepared?", top_k=2)

    mock_embedder.encode.assert_called_once_with("How is the data prepared?")
    mock_client.query_points.assert_called_once_with(
        collection_name="devwhisper",
        query=vector,
        limit=2,
        score_threshold=0.0,
    )

    assert "Result 1:" in context
    assert "File: pipeline.py" in context
    assert "Function: preprocess" in context
    assert "Start Line: 12" in context
    assert "Result 2:" in context
    assert "Function: train_model" in context


def test_retrieve_returns_empty_string_when_no_matches(monkeypatch):
    """An empty Qdrant response produces an empty context string."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.0]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = []

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    assert retriever.retrieve("missing symbol") == ""


def test_retrieve_uses_safe_defaults_for_missing_payload_fields(monkeypatch):
    """Incomplete point payloads are formatted without raising errors."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.4]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [_point({})]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("unknown code")

    assert "File: unknown" in context
    assert "Function: unknown" in context
    assert "Start Line: ?" in context
    assert "Code:\n\n" in context


def test_retrieve_handles_none_payload(monkeypatch):
    """A point whose payload is None is treated like an empty payload."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.5]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [_point(None)]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("unstructured point")

    assert "Result 1:" in context
    assert "File: unknown" in context
    assert "Function: unknown" in context


def test_retrieve_marks_non_function_snippet_as_unknown(monkeypatch):
    """Snippets without a regular ``def`` line keep the fallback name."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.6]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point(
            {
                "file": "settings.py",
                "start_line": 1,
                "text": "DEBUG = False\nTIMEOUT = 30",
            }
        )
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("Where is timeout configured?")

    assert "File: settings.py" in context
    assert "Function: unknown" in context
    assert "DEBUG = False" in context


def test_retrieve_with_sources_returns_tuple_and_deduplicates(monkeypatch):
    """Return unique source files in order when sources are requested."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.7]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "text": "def a(): pass"}),
        _point({"file": "b.py", "text": "def b(): pass"}),
        _point({"file": "a.py", "text": "def a2(): pass"}),
        _point({"file": "unknown", "text": "def unknown(): pass"}),
        _point({"file": None, "text": "def empty(): pass"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context, sources = retriever.retrieve("test query", include_sources=True)

    assert isinstance(context, str)
    assert "File: a.py" in context
    assert "File: b.py" in context
    assert sources == ["a.py", "b.py"]


def test_retrieve_deduplicates_exact_duplicates(monkeypatch):
    """Exact duplicate chunks are deduplicated, preserving the highest-ranked."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "start_line": 10, "text": "def foo():\n    return 42"}),
        _point({"file": "a.py", "start_line": 10, "text": "def foo():\n    return 42"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=2)

    assert "Result 1:" in context
    assert "Result 2:" not in context
    assert "File: a.py" in context
    assert "Start Line: 10" in context


def test_retrieve_deduplicates_whitespace_minor_differences(monkeypatch):
    """Chunks with only minor whitespace differences are deduplicated."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "start_line": 10, "text": "def foo():\n    return 42"}),
        _point({"file": "a.py", "start_line": 10, "text": "def foo(   ):\n\treturn    42"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=2)

    assert "Result 1:" in context
    assert "Result 2:" not in context


def test_retrieve_deduplicates_overlapping_same_file(monkeypatch):
    """Overlapping chunks in the same file (by substring or line ranges) are deduplicated."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        # Chunk 1: range 10-12
        _point({"file": "a.py", "start_line": 10, "text": "line 10\nline 11\nline 12"}),
        # Chunk 2: range 12-14 (overlaps at line 12)
        _point({"file": "a.py", "start_line": 12, "text": "line 12\nline 13\nline 14"}),
        # Chunk 3: substring containment
        _point({"file": "a.py", "start_line": 10, "text": "line 10\nline 11"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=3)

    assert "Result 1:" in context
    assert "Result 2:" not in context
    assert "Result 3:" not in context


def test_retrieve_deduplicates_across_different_files(monkeypatch):
    """Exact duplicate content in different files is deduplicated."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "start_line": 10, "text": "def shared_logic():\n    pass"}),
        _point({"file": "b.py", "start_line": 20, "text": "def shared_logic():\n    pass"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=2)

    assert "Result 1:" in context
    assert "File: a.py" in context
    assert "Result 2:" not in context
    assert "File: b.py" not in context


def test_retrieve_deduplicates_different_metadata(monkeypatch):
    """Identical content with different metadata is deduplicated."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "start_line": 10, "text": "def test():\n    pass"}),
        _point({"file": "a.py", "start_line": 99, "text": "def test():\n    pass"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=2)

    assert "Result 1:" in context
    assert "Start Line: 10" in context
    assert "Result 2:" not in context


def test_retrieve_deduplicates_fuzzy_similarity(monkeypatch):
    """Chunks with high similarity (SequenceMatcher ratio >= 0.85) are deduplicated."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "text": "def compute():\n    a = 1\n    b = 2\n    return a + b"}),
        _point({"file": "a.py", "text": "def compute():\n    a = 1\n    b = 3\n    return a + b"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=2)

    assert "Result 1:" in context
    assert "Result 2:" not in context


def test_retrieve_empty_results_and_single_chunk(monkeypatch):
    """Retrieval handles empty lists and single-chunk lists cleanly."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()

    # 1. Empty results
    mock_client.query_points.return_value.points = []
    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    assert retriever.retrieve("test query") == ""

    # 2. Single chunk
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "text": "def one(): pass"})
    ]
    context = retriever.retrieve("test query")
    assert "Result 1:" in context
    assert "File: a.py" in context
    assert "Result 2:" not in context


def test_retrieve_preserves_unique_chunks_and_ranking_order(monkeypatch):
    """Unique chunks are preserved and their relative ranking order is maintained."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "first.py", "text": "def first_rank(): pass"}),
        _point({"file": "second.py", "text": "def second_rank(): pass"}),
        _point({"file": "third.py", "text": "def third_rank(): pass"}),
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=3)

    assert "Result 1:" in context
    assert "File: first.py" in context
    assert "Result 2:" in context
    assert "File: second.py" in context
    assert "Result 3:" in context
    assert "File: third.py" in context

    # Check order in the context string
    idx1 = context.find("File: first.py")
    idx2 = context.find("File: second.py")
    idx3 = context.find("File: third.py")
    assert idx1 < idx2 < idx3