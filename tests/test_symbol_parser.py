"""Unit tests for the AST-based symbol parser."""

import tempfile
import os
import sys
import pytest

from symbol_parser import (
    extract_symbols_from_source,
    extract_symbols_from_file,
    Symbol,
)


# --- extract_symbols_from_source ---

def test_extracts_top_level_function():
    source = '''def preprocess(data):
    """Clean the input data."""
    return data.dropna()
'''
    symbols = extract_symbols_from_source(source)
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "preprocess"
    assert sym.symbol_type == "function"
    assert sym.start_line == 1
    assert sym.docstring == "Clean the input data."
    assert sym.parent_class is None
    assert "def preprocess(data):" in sym.source


def test_extracts_async_function():
    source = '''async def fetch(url):
    """Fetch a URL."""
    return await http_get(url)
'''
    symbols = extract_symbols_from_source(source)
    assert len(symbols) == 1
    assert symbols[0].name == "fetch"
    assert symbols[0].symbol_type == "function"
    assert "async def fetch(url):" in symbols[0].source


def test_extracts_class_and_methods():
    source = '''class DataProcessor:
    """Process raw data."""

    def __init__(self, path):
        self.path = path

    def load(self):
        """Load data from disk."""
        return open(self.path).read()
'''
    symbols = extract_symbols_from_source(source)
    names = {s.name for s in symbols}
    assert names == {"DataProcessor", "__init__", "load"}

    class_sym = next(s for s in symbols if s.name == "DataProcessor")
    assert class_sym.symbol_type == "class"
    assert class_sym.docstring == "Process raw data."

    method_sym = next(s for s in symbols if s.name == "load")
    assert method_sym.symbol_type == "method"
    assert method_sym.parent_class == "DataProcessor"
    assert method_sym.docstring == "Load data from disk."


def test_extracts_decorated_function():
    source = '''@property
def name(self):
    """Return the name."""
    return self._name
'''
    symbols = extract_symbols_from_source(source)
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "name"
    assert sym.start_line == 1  # decorator line
    assert "@property" in sym.source
    assert "def name(self):" in sym.source


def test_handles_nested_class():
    source = '''class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def method(self):
            pass
'''
    symbols = extract_symbols_from_source(source)
    names = [s.name for s in symbols]
    assert names == ["Outer", "Inner", "method"]

    inner = next(s for s in symbols if s.name == "Inner")
    assert inner.symbol_type == "class"

    method = next(s for s in symbols if s.name == "method")
    assert method.symbol_type == "method"
    assert method.parent_class == "Inner"


def test_handles_empty_source():
    assert extract_symbols_from_source("") == []


def test_handles_syntax_error():
    source = "def broken(\n"
    assert extract_symbols_from_source(source) == []


def test_preserves_line_endings_in_source():
    source = "def foo():\n    pass\n"
    symbols = extract_symbols_from_source(source)
    assert symbols[0].source == source


# --- extract_symbols_from_file ---

def test_reads_real_file():
    source = '''def helper(x):
    return x * 2
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        tmp_path = f.name

    try:
        symbols = extract_symbols_from_file(tmp_path)
        assert len(symbols) == 1
        assert symbols[0].name == "helper"
    finally:
        os.unlink(tmp_path)


def test_returns_empty_for_missing_file():
    assert extract_symbols_from_file("/nonexistent/path/file.py") == []


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not support disabling read permissions via chmod")
def test_returns_empty_for_unreadable_file(tmp_path):
    bad = tmp_path / "unreadable.py"
    bad.write_text("def foo(): pass")
    os.chmod(bad, 0o000)
    try:
        assert extract_symbols_from_file(str(bad)) == []
    finally:
        os.chmod(bad, 0o644)


def test_extracts_js_ts_functions_and_classes():
    js_code = """
export function calculateSum(a, b) {
    return a + b;
}

const fetchData = async () => {
    return fetch('/api');
};

class AuthController {
    constructor() {}
}
"""
    symbols = extract_symbols_from_source(js_code, filename="controller.ts")
    names = {s.name for s in symbols}
    assert "calculateSum" in names
    assert "fetchData" in names
    assert "AuthController" in names
    
    cls_sym = next(s for s in symbols if s.name == "AuthController")
    assert cls_sym.symbol_type == "class"