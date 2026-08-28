import pytest
from query_normalizer import QueryNormalizer, extract_query_filters

def test_extract_query_filters_basic():
    qn = QueryNormalizer()
    clean_q, filters = qn.extract_filters("How does preprocess work file:indexer.py type:function")
    assert clean_q == "How does preprocess work"
    assert filters == {"file": "indexer.py", "symbol_type": "function"}

def test_extract_query_filters_convenience_function():
    clean_q, filters = extract_query_filters("what is Symbol class symbol:Symbol")
    assert clean_q == "what is Symbol class"
    assert filters == {"symbol_name": "Symbol"}

def test_extract_query_filters_empty():
    clean_q, filters = extract_query_filters("")
    assert clean_q == ""
    assert filters == {}

def test_retriever_query_filters_integration(monkeypatch):
    import retriever
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    mock_client = MagicMock()
    mock_client.query_points.return_value = SimpleNamespace(points=[SimpleNamespace(payload={"file": "main.py", "start_line": 1, "text": "def test(): pass"})])

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    # Calling retrieve with inline filter
    res = retriever.retrieve("what does preprocess do? file:main.py", include_sources=True)
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert mock_client.query_points.called
