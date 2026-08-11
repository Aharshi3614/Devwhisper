"""Tests for Query Normalization Layer (Issue #218)."""
import pytest
from query_normalizer import QueryNormalizer, normalize_query

def test_normalize_whitespace():
    qn = QueryNormalizer()
    assert qn.normalize_whitespace("  hello   world  \n") == "hello world"

def test_normalize_punctuation():
    qn = QueryNormalizer()
    assert qn.normalize_punctuation('"hello world!"') == "hello world"

def test_normalize_capitalization():
    qn = QueryNormalizer()
    assert qn.normalize_capitalization("WHERE IS get_user_data") == "where is get_user_data"

def test_full_query_normalization():
    res = normalize_query("   WHAT DOES   preprocess_query DO???   ")
    assert "preprocess_query" in res
    assert "what does" in res
