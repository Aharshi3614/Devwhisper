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


import re


def _extract_js_ts_symbols(source: str) -> List[Symbol]:
    """
    Extract functions, async functions, arrow functions, and classes from JS/TS source.
    """
    symbols: List[Symbol] = []
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    # Patterns for JS/TS declarations
    func_pattern = re.compile(r'^(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_$]+)\s*\(')
    class_pattern = re.compile(r'^(?:export\s+)?class\s+([a-zA-Z0-9_$]+)')
    arrow_pattern = re.compile(r'^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_$]+)\s*=>')

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        line_no = idx + 1

        # Check function
        m_func = func_pattern.match(line)
        if m_func:
            name = m_func.group(1)
            # Find end by matching braces or approximation
            end_line = min(len(lines), line_no + 20)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="function",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    parent_class=None,
                )
            )
            continue

        # Check class
        m_class = class_pattern.match(line)
        if m_class:
            name = m_class.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="class",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    parent_class=None,
                )
            )
            continue

        # Check const arrow function
        m_arrow = arrow_pattern.match(line)
        if m_arrow:
            name = m_arrow.group(1)
            end_line = min(len(lines), line_no + 20)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="function",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    parent_class=None,
                )
            )

    return symbols


def extract_symbols_from_source(source: str, filename: str = "<unknown>") -> List[Symbol]:
    """Parse *source* and return all extractable symbols for Python or JS/TS files.

    Args:
        source: Full source code.
        filename: Used for format detection and error reporting.

    Returns:
        A list of :class:`Symbol` objects. Empty on syntax errors.
    """
    ext = Path(filename).suffix.lower()
    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
        return _extract_js_ts_symbols(source)

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
        filepath: Path to a supported source file (.py, .js, .jsx, .ts, .tsx).

    Returns:
        A list of :class:`Symbol` objects. Empty on unreadable files or syntax errors.
    """
    path = Path(filepath)
    if not path.is_file():
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    return extract_symbols_from_source(source, filename=str(path))
