"""Unit tests for the semantic code retriever."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import retriever


def _embedding(values):
    """Return an object matching SentenceTransformer.encode() output."""
    encoded = MagicMock()
    encoded.tolist.return_value = values
    return encoded


def test_retrieve_sends_list_embedding_with_expected_shape(monkeypatch):
    """The encoded query should become a 384-value list for Qdrant."""
    vector = [0.25] * 384
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = _embedding(vector)

    mock_client = MagicMock()
    mock_client.query_points.return_value = SimpleNamespace(points=[])

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    result = retriever.retrieve("Where is authentication handled?", top_k=4)

    assert result == ""
    mock_embedder.encode.assert_called_once_with(
        "Where is authentication handled?"
    )
    mock_client.query_points.assert_called_once_with(
        collection_name="devwhisper",
        query=vector,
        limit=4,
    )

    sent_vector = mock_client.query_points.call_args.kwargs["query"]
    assert isinstance(sent_vector, list)
    assert len(sent_vector) == 384
    assert all(isinstance(value, float) for value in sent_vector)


def test_retrieve_handles_empty_query(monkeypatch):
    """An empty query should be encoded and handled without a live service."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = _embedding([0.0] * 384)

    mock_client = MagicMock()
    mock_client.query_points.return_value = SimpleNamespace(points=[])

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    result = retriever.retrieve("")

    assert result == ""
    mock_embedder.encode.assert_called_once_with("")
    mock_client.query_points.assert_called_once()


def test_retrieve_formats_mocked_qdrant_results(monkeypatch):
    """Qdrant payloads should be converted into readable ranked context."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = _embedding([0.1] * 384)

    points = [
        SimpleNamespace(
            payload={
                "file": "services/auth.py",
                "start_line": 18,
                "text": (
                    "def authenticate_user(username, password):\n"
                    "    return username == 'admin'"
                ),
            }
        ),
        SimpleNamespace(
            payload={
                "file": "utils/constants.py",
                "start_line": 3,
                "text": "TOKEN_TTL_SECONDS = 3600",
            }
        ),
    ]
    mock_client = MagicMock()
    mock_client.query_points.return_value = SimpleNamespace(points=points)

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    result = retriever.retrieve("How does authentication work?", top_k=2)

    assert "Result 1:" in result
    assert "File: services/auth.py" in result
    assert "Function: authenticate_user" in result
    assert "Start Line: 18" in result
    assert "return username == 'admin'" in result

    assert "Result 2:" in result
    assert "File: utils/constants.py" in result
    assert "Function: unknown" in result
    assert "Start Line: 3" in result
    assert "TOKEN_TTL_SECONDS = 3600" in result


def test_retrieve_returns_empty_string_for_no_matches(monkeypatch):
    """No Qdrant matches should produce an empty context string."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = _embedding([0.5] * 384)

    mock_client = MagicMock()
    mock_client.query_points.return_value = SimpleNamespace(points=[])

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    assert retriever.retrieve("unmatched query", top_k=3) == ""
    context = retriever.retrieve("Where is timeout configured?")

    assert "File: settings.py" in context
    assert "Function: unknown" in context
    assert "DEBUG = False" in context


def test_retrieve_with_sources_returns_tuple_and_deduplicates(monkeypatch):
    """Retrieve with include_sources=True returns unique files in order, ignoring 'unknown'."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.7]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _point({"file": "a.py", "text": "def a(): pass"}),
        _point({"file": "b.py", "text": "def b(): pass"}),
        _point({"file": "a.py", "text": "def a2(): pass"}),  # duplicate
        _point({"file": "unknown", "text": "def unknown(): pass"}),  # unknown
        _point({"file": None, "text": "def empty(): pass"}),  # none/empty
    ]

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    context, sources = retriever.retrieve("test query", include_sources=True)

    assert isinstance(context, str)
    assert "File: a.py" in context
    assert "File: b.py" in context
    assert sources == ["a.py", "b.py"]
