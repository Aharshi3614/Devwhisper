"""Unit tests for the Qdrant-backed code retrieval formatter."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import retriever
import vector_store


def _point(payload):
    """Build a lightweight object matching Qdrant's returned point shape."""
    return SimpleNamespace(payload=payload)


def _setup_mocks(monkeypatch, points):
    """Helper to mock embedder and vector_store queries consistently across tests."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = points

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)
    monkeypatch.setattr(vector_store, "client", mock_client)

    return mock_embedder, mock_client


def test_retrieve_formats_ranked_results(monkeypatch):
    """Relevant Qdrant points are returned as readable, ranked context."""
    points = [
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
    mock_embedder, mock_client = _setup_mocks(monkeypatch, points)

    context = retriever.retrieve("How is the data prepared?", top_k=2)

    mock_embedder.encode.assert_called_once_with("How is the data prepared")
    mock_client.query_points.assert_called_once_with(
        collection_name="devwhisper",
        query=[0.1, 0.2, 0.3],
        query_filter=None,
        limit=2,
        score_threshold=0.0,
    )

    assert "Result 1:" in context
    assert "File: pipeline.py" in context
    assert "Function: preprocess" in context
    assert "Location: Line 12" in context
    assert "Result 2:" in context
    assert "Function: train_model" in context


def test_retrieve_returns_empty_string_when_no_matches(monkeypatch):
    """An empty Qdrant response produces an empty context string."""
    _setup_mocks(monkeypatch, [])
    assert retriever.retrieve("missing symbol") == ""


def test_retrieve_uses_safe_defaults_for_missing_payload_fields(monkeypatch):
    """Incomplete point payloads are formatted without raising errors."""
    _setup_mocks(monkeypatch, [_point({})])
    context = retriever.retrieve("unknown code")

    assert "File: unknown" in context
    assert "Function: unknown" in context
    assert "Location: Line ?" in context
    assert "Code:\n\n" in context


def test_retrieve_handles_none_payload(monkeypatch):
    """A point whose payload is None is treated like an empty payload."""
    _setup_mocks(monkeypatch, [_point(None)])
    context = retriever.retrieve("unstructured point")

    assert "Result 1:" in context
    assert "File: unknown" in context
    assert "Function: unknown" in context


def test_retrieve_marks_non_function_snippet_as_unknown(monkeypatch):
    """Snippets without a regular ``def`` line keep the fallback name."""
    points = [
        _point(
            {
                "file": "settings.py",
                "start_line": 1,
                "text": "DEBUG = False\nTIMEOUT = 30",
            }
        )
    ]
    _setup_mocks(monkeypatch, points)
    context = retriever.retrieve("Where is timeout configured?")

    assert "File: settings.py" in context
    assert "Function: unknown" in context
    assert "DEBUG = False" in context


def test_retrieve_with_sources_returns_tuple_and_deduplicates(monkeypatch):
    """Return unique source files in order when sources are requested."""
    points = [
        _point({"file": "a.py", "text": "def a(): pass"}),
        _point({"file": "b.py", "text": "def b(): pass"}),
        _point({"file": "a.py", "text": "def a2(): pass"}),
        _point({"file": "unknown", "text": "def unknown(): pass"}),
        _point({"file": None, "text": "def empty(): pass"}),
    ]
    _setup_mocks(monkeypatch, points)
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
    monkeypatch.setattr(retriever, "_get_bm25", lambda repo_id: None)
    points = [_point({"file": "test.py", "start_line": 1, "text": "def foo(): pass"})]
    _setup_mocks(monkeypatch, points)

    context = retriever.retrieve("test query", top_k=2)
    assert "Result 1:" in context
    assert "File: test.py" in context


def test_exact_symbol_search_prefers_metadata_match(monkeypatch):
    """Symbol chunks with matching symbol_name get exact metadata hits."""
    monkeypatch.setattr(
        retriever, "_get_bm25",lambda repo_id:
        {
            "bm25": MagicMock(),
            "chunks": [
                {"text": "def preprocess(data): pass", "symbol_name": "preprocess", "is_symbol": True},
                {"text": "def preprocess_data(x): pass", "symbol_name": "preprocess_data", "is_symbol": True},
                {"text": "some random text", "is_symbol": False},
            ],
            "corpus": ["def preprocess(data): pass", "def preprocess_data(x): pass", "some random text"],
        }
    )
    results = retriever._exact_symbol_search(["preprocess"], top_k=5)
    names = [r.get("symbol_name") for r in results]
    assert "preprocess" in names
    assert "preprocess_data" not in names


def test_retrieve_shows_method_with_parent_class(monkeypatch):
    """Method symbols display as ClassName.method_name."""
    points = [
        _point(
            {
                "file": "model.py",
                "start_line": 8,
                "end_line": 10,
                "text": "    def train(self):\n        pass",
                "symbol_name": "train",
                "symbol_type": "method",
                "parent_class": "Model",
            }
        ),
    ]
    _setup_mocks(monkeypatch, points)

    context = retriever.retrieve("How do I train the model?", top_k=1)

    assert "Method: Model.train" in context
    assert "Location: Lines 8-10" in context


def test_retrieve_falls_back_to_regex_for_line_chunks(monkeypatch):
    """Non-symbol chunks still use the old def-line regex fallback."""
    points = [
        _point(
            {
                "file": "utils.py",
                "start_line": 3,
                "text": "def helper():\n    pass",
                "is_symbol": False,
            }
        ),
    ]
    _setup_mocks(monkeypatch, points)

    context = retriever.retrieve("What is helper?", top_k=1)

    assert "Function: helper" in context
    assert "Location: Line 3" in context


# ── repository tag filtering (shared-index mode, from PR #212) ────────────

def test_keyword_search_filters_by_repository(monkeypatch):
    """BM25 chunks tagged with a different repository are excluded."""
    bm25 = MagicMock()
    bm25.get_scores.return_value = [0.5, 0.5]
    monkeypatch.setattr(
        retriever, "_get_bm25", lambda repo_id: {
            "bm25": bm25,
            "chunks": [
                {"text": "def foo():\n    pass", "repository": "repoA"},
                {"text": "def foo():\n    pass", "repository": "repoB"},
            ],
            "corpus": ["def foo():\n    pass", "def foo():\n    pass"],
        }
    )

    results = retriever._keyword_search("foo", top_k=10, repository_names=["repoA"])
    repos = {c["repository"] for c in results}
    assert repos == {"repoA"}


def test_exact_symbol_search_filters_by_repository(monkeypatch):
    """Symbol chunks tagged with a different repository are excluded."""
    monkeypatch.setattr(
        retriever, "_get_bm25", lambda repo_id: {
            "bm25": MagicMock(),
            "chunks": [
                {"text": "def helper():\n    pass", "symbol_name": "helper", "is_symbol": True, "repository": "repoA"},
                {"text": "def helper():\n    pass", "symbol_name": "helper", "is_symbol": True, "repository": "repoB"},
            ],
            "corpus": ["def helper():\n    pass", "def helper():\n    pass"],
        }
    )

    results = retriever._exact_symbol_search(["helper"], top_k=5, repository_names=["repoA"])
    repos = {c["repository"] for c in results}
    assert repos == {"repoA"}
