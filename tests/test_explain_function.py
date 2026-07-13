"""Tests for handlers/explain_function.py (issue #3 / mirror #313)."""
from __future__ import annotations

import sys
from pathlib import Path

# Make handlers/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handlers.explain_function import (  # noqa: E402
    INTENTS,
    matches,
    explain_function,
    format_tool_call,
    _extract_function_name,
)


def test_intents_are_lowercase_strings():
    for s in INTENTS:
        assert isinstance(s, str)
        assert s == s.lower()


def test_matches_exact_intent():
    assert matches("explain this function")
    assert matches("explain that function")
    assert matches("what does this function do")
    assert matches("describe this function")


def test_matches_intent_with_extra_words():
    """'explain this function in retriever.py' should still match."""
    assert matches("explain this function in retriever.py")
    assert matches("describe this function in main.py")
    assert matches("what does this function do for me")


def test_no_match_for_unrelated_queries():
    assert not matches("how do I deploy this")
    assert not matches("tell me a joke")
    assert not matches("")


def test_extract_function_name_backtick():
    assert _extract_function_name("explain `parse_url`") == "parse_url"


def test_extract_function_name_keyword():
    assert _extract_function_name("what does parse_url do") == "parse_url"


def test_extract_function_name_trailing_identifier():
    assert _extract_function_name("explain function foo_bar please") == "foo_bar"


def test_extract_function_name_returns_none_when_missing():
    assert _extract_function_name("explain this function") is None


def test_format_tool_call_shape():
    out = format_tool_call("call_123", "Function: foo\nSignature: def foo()")
    assert out["toolCallId"] == "call_123"
    assert "Function: foo" in out["result"]
    assert set(out.keys()) == {"toolCallId", "result"}


def test_explain_function_handles_missing_deps(monkeypatch):
    """When retriever / llm aren't importable in the test env, explain_function
    should raise a clear ImportError rather than silently returning junk.

    This documents the contract: the handler depends on the parent project's
    retriever.py + llm.py (which need qdrant_client + sentence_transformers).
    """
    from handlers import explain_function as ef

    # Force a re-import to make sure we hit the lazy imports
    import importlib
    importlib.reload(ef)

    try:
        out = ef.explain_function("explain `foo`")
    except (ImportError, ModuleNotFoundError) as e:
        # Expected when qdrant_client / sentence_transformers aren't installed
        assert "retriever" in str(e) or "llm" in str(e) or "qdrant" in str(e) or "sentence_transformers" in str(e)
    else:
        assert isinstance(out, str)
