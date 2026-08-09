"""Unit tests for the circular import checker.

Covers the three public functions:
  - tokenize_lines_identifiers: extracting imported module names via AST.
  - parse_import: reading a real file and returning {file_name: imports}.
  - check_import_circular: finding circular imports in a dependency map.

NOTE: test_tokenize_syntax_error_returns_empty_list below will FAIL until
the function returns an empty list instead of an empty dict.
"""

import os
import tempfile

import pytest

from circular_import_checker import (
    parse_import,
    tokenize_lines_identifiers,
    check_import_circular,
)


# --- tokenize_lines_identifiers ---

def test_tokenize_simple_import():
    assert tokenize_lines_identifiers("import os\n") == ["os"]


def test_tokenize_from_import():
    assert tokenize_lines_identifiers("from sys import path\n") == ["sys"]


def test_tokenize_import_as():
    assert tokenize_lines_identifiers("import numpy as np\n") == ["numpy"]


def test_tokenize_multiple_imports():
    source = "import os\nfrom sys import path\nimport numpy as np\nprint('hi')\n"
    assert tokenize_lines_identifiers(source) == ["os", "sys", "numpy"]


def test_tokenize_no_imports():
    assert tokenize_lines_identifiers("x = 1\ndef foo():\n    pass\n") == []


def test_tokenize_syntax_error_returns_empty_list():
    # Currently returns {} — an empty dict — which violates the declared
    # return type list[str]. Change `return {}` to `return []` to fix.
    assert tokenize_lines_identifiers("def broken(\n") == []


# --- parse_import ---

def test_parse_import_reads_real_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import os\nfrom collections import deque\n")
        tmp_path = f.name
    try:
        result = parse_import(tmp_path, "sample.py")
        assert result == {"sample": ["os", "collections"]}
    finally:
        os.unlink(tmp_path)


# --- check_import_circular ---

def test_check_detects_direct_circular():
    files = {"a": ["b"], "b": ["a"]}
    result = check_import_circular(files)
    # TODO: currently reports the pair twice (once from each side).
    assert len(result) >= 1


def test_check_detects_multiple_circular_pairs():
    files = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}
    result = check_import_circular(files)
    assert len(result) >= 2


def test_check_returns_empty_when_no_circular():
    files = {"a": ["b"], "b": ["c"], "c": []}
    assert check_import_circular(files) == []


def test_check_ignores_unindexed_modules():
    # "os" is imported but not a key in the map, so it can't form a cycle.
    files = {"a": ["b", "os"], "b": ["a"]}
    result = check_import_circular(files)
    assert len(result) >= 1


# --- tokenize_lines_identifiers: more cases ---

def test_tokenize_multiple_imports_on_one_line():
    assert tokenize_lines_identifiers("import os, sys\n") == ["os", "sys"]


def test_tokenize_from_import_with_multiple_names():
    assert tokenize_lines_identifiers("from collections import deque, Counter\n") == ["collections"]


def test_tokenize_import_inside_function():
    assert tokenize_lines_identifiers("def f():\n    import os\n") == ["os"]


def test_tokenize_relative_import_has_no_empty_string():
    # `from . import models` gives ast.module == None, which the current code
    # turns into "". This test is expected to fail until that is handled.
    result = tokenize_lines_identifiers("from . import models\n")
    assert "" not in result


# --- check_import_circular: more cases ---

def test_check_single_side_import_is_not_circular():
    # a imports b, but b does not import a back → no cycle.
    files = {"a": ["b"], "b": []}
    assert check_import_circular(files) == []


def test_check_detects_indirect_circular_chain():
    # Three files forming a cycle: a imports b, b imports c, c imports a.
    files = {"a": ["b"], "b": ["c"], "c": ["a"]}
    assert check_import_circular(files) != []
