"""Tests for the retrieval confidence indicator.

`retrieve()` now returns a confidence table alongside the formatted context
and sources: a mapping from source label to a rounded confidence percentage
(0-100), or ``None`` when no similarity score is available.

The `/stream` endpoint renders each source with its confidence percentage and
marks low-confidence sources (below 50%) with a warning emoji, so users can
tell which results the retrieval system is not sure about.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import retriever
from main import _sources_display


def _point(payload, score=None):
    """Build a Qdrant-like point with an optional similarity score.

    Real Qdrant points carry a ``score`` attribute that is only present at
    query time. Tests pass it explicitly, or omit it to simulate sources that
    have no similarity score (e.g. BM25-only matches).
    """
    return SimpleNamespace(payload=payload, score=score)


def _setup_mocks(monkeypatch, points):
    """Mock the embedder and vector store the same way test_retriever.py does."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = points

    monkeypatch.setattr(retriever, "embedder", mock_embedder)
    monkeypatch.setattr(retriever, "client", mock_client)

    import vector_store
    monkeypatch.setattr(vector_store, "client", mock_client)

    return mock_embedder, mock_client


def test_retrieve_returns_confidence_table(monkeypatch):
    """Confidence is a rounded percentage per source, or None without a score."""
    points = [
        _point({"file": "a.py", "text": "def a(): pass"}, score=0.92),
        _point({"file": "b.py", "text": "def b(): pass"}, score=0.30),
        _point({"file": "c.py", "text": "def c(): pass"}),  # no similarity score
    ]
    _setup_mocks(monkeypatch, points)

    context, sources, confidences = retriever.retrieve("test query", include_sources=True)

    assert isinstance(confidences, dict)
    assert confidences["a.py"] == 92
    assert confidences["b.py"] == 30
    assert confidences["c.py"] is None


def test_confidence_rounds_to_nearest_percent(monkeypatch):
    """Scores are multiplied by 100 and rounded, not truncated."""
    points = [
        _point({"file": "a.py", "text": "def a(): pass"}, score=0.9264),
        _point({"file": "b.py", "text": "def b(): pass"}, score=0.6103),
    ]
    _setup_mocks(monkeypatch, points)

    _, _, confidences = retriever.retrieve("test query", include_sources=True)

    assert confidences["a.py"] == 93   # round(92.64) = 93
    assert confidences["b.py"] == 61   # round(61.03) = 61


def test_sources_display_shows_confidence_percentages():
    """Sources render with their percentage; missing scores render plain."""
    text = _sources_display(
        ["a.py", "b.py", "c.py"],
        {"a.py": 92, "b.py": 30, "c.py": None},
    )

    assert "**Sources used:**" in text
    assert "`a.py` (92%)" in text
    assert "`c.py`" in text
    assert "`c.py` (None%)" not in text  # no fake percentage for missing scores


def test_sources_display_marks_low_confidence_with_warning():
    """Sources below the 50% threshold get an explicit warning."""
    text = _sources_display(["b.py"], {"b.py": 30})

    assert "`b.py` (30% ⚠️)" in text


def test_sources_display_does_not_warn_above_threshold():
    """Sources at or above 50% are shown without a warning."""
    text = _sources_display(["ok.py", "edge.py"], {"ok.py": 50, "edge.py": 87})

    assert "`ok.py` (50%)" in text
    assert "`edge.py` (87%)" in text
    assert "⚠️" not in text
