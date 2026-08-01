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
        query_filter=None,
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


def test_rrf_fusion_ranks_shared_docs_higher():
    """Document appearing in multiple lists gets a higher RRF rank."""
    list_a = [{"_idx": 1}, {"_idx": 2}, {"_idx": 3}]
    list_b = [{"_idx": 2}, {"_idx": 4}, {"_idx": 5}]
    fused = retriever._rrf_fusion([list_a, list_b], k=60, final_top_k=5)
    indices = [d["_idx"] for d in fused]
    assert indices[0] == 2, f"Expected shared doc (idx=2) first, got {indices}"


def test_extract_symbols_finds_function_names():
    symbols = retriever._extract_symbols("What does the retrieve() function do?")
    assert "retrieve" in symbols


def test_extract_symbols_finds_camel_case():
    symbols = retriever._extract_symbols("Where is the DataProcessor class?")
    assert "DataProcessor" in symbols


def test_hybrid_retrieve_falls_back_to_vector_only(monkeypatch):
    """When BM25 index is absent, degrade to pure vector search."""
    monkeypatch.setattr(retriever, "_bm25_data", None)

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.0]
    monkeypatch.setattr(retriever, "embedder", mock_embedder)

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "test.py", "start_line": 1, "text": "def foo(): pass"})
    ]
    monkeypatch.setattr(retriever, "client", mock_client)

    context = retriever.retrieve("test query", top_k=2)
    assert "Result 1:" in context
    assert "File: test.py" in context


def test_retrieve_with_metadata_filter(monkeypatch):
    """Retrieval passes metadata filters into Qdrant query_points."""
    monkeypatch.setattr(retriever, "_bm25_data", None)

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    monkeypatch.setattr(retriever, "embedder", mock_embedder)

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "model.py", "start_line": 5, "text": "def train(): pass"})
    ]
    monkeypatch.setattr(retriever, "client", mock_client)

    metadata_filter = {"file": "model.py"}
    context = retriever.retrieve("train model", metadata_filter=metadata_filter)

    assert "File: model.py" in context
    _, kwargs = mock_client.query_points.call_args
    assert kwargs["query_filter"] is not None
    assert kwargs["query_filter"].must[0].key == "file"
    assert kwargs["query_filter"].must[0].match.value == "model.py"


def test_matches_metadata_filter_helper():
    """Helper evaluates payload attributes against metadata query filters."""
    payload = {"file": "indexer.py", "language": "python"}
    assert retriever._matches_metadata_filter(payload, {"file": "indexer.py"}) is True
    assert retriever._matches_metadata_filter(payload, {"file": "main.py"}) is False
    assert retriever._matches_metadata_filter(payload, None) is True
    