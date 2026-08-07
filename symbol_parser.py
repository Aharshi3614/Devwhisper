"""AST-based symbol extraction for Python source files.

Uses the built-in :mod:`ast` module to extract functions, classes, and
methods as discrete indexing units.  Each symbol carries line-number
metadata so the indexer can embed and store it independently of
line-based chunks.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Symbol:
    """A logical code entity extracted from a Python file."""

    name: str
    symbol_type: str  # "function", "class", or "method"
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str]
    parent_class: Optional[str] = None


class _SymbolExtractor(ast.NodeVisitor):
    """Walk an AST and collect top-level and nested symbols."""

    def __init__(self, source_lines: List[str]) -> None:
        self.source_lines = source_lines
        self.symbols: List[Symbol] = []
        self._current_class: Optional[str] = None

    def _get_source(self, start: int, end: int) -> str:
        """Return the raw source text between *start* and *end* (1-based, inclusive)."""
        return "".join(self.source_lines[start - 1 : end])

    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Return the docstring body if the node has one, else None."""
        doc = ast.get_docstring(node)  # type: ignore[arg-type]
        return doc

    def _start_line(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
        """Return the first line of the node, accounting for decorators."""
        if node.decorator_list:
            return min(d.lineno for d in node.decorator_list)
        return getattr(node, "lineno", 1)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Common logic for sync and async functions."""
        start = self._start_line(node)
        end = getattr(node, "end_lineno", start)

        symbol_type = "method" if self._current_class else "function"

        symbol = Symbol(
            name=node.name,
            symbol_type=symbol_type,
            start_line=start,
            end_line=end,
            source=self._get_source(start, end),
            docstring=self._extract_docstring(node),
            parent_class=self._current_class,
        )
        self.symbols.append(symbol)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                old_class = self._current_class
                self._current_class = node.name
                self._visit_function(child)
                self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start = self._start_line(node)
        end = getattr(node, "end_lineno", start)

        symbol = Symbol(
            name=node.name,
            symbol_type="class",
            start_line=start,
            end_line=end,
            source=self._get_source(start, end),
            docstring=self._extract_docstring(node),
            parent_class=None,
        )
        self.symbols.append(symbol)

        old_class = self._current_class
        self._current_class = node.name

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_function(child)
            elif isinstance(child, ast.ClassDef):
                self.visit_ClassDef(child)

        self._current_class = old_class


def extract_symbols_from_source(source: str, filename: str = "<unknown>") -> List[Symbol]:
    """Parse *source* and return all extractable symbols.

    Args:
        source: Full Python source code.
        filename: Used only for error reporting.

    Returns:
        A list of :class:`Symbol` objects.  Empty on syntax errors.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [""]

    extractor = _SymbolExtractor(lines)
    extractor.visit(tree)
    return extractor.symbols


def extract_symbols_from_file(filepath: str) -> List[Symbol]:
    """Read *filepath* and return all extractable symbols.

    Args:
        filepath: Path to a Python ``.py`` file.

    Returns:
        A list of :class:`Symbol` objects.  Empty on unreadable files or
        syntax errors.
    """
    path = Path(filepath)
    if not path.is_file():
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    return extract_symbols_from_source(source, filename=str(path))
