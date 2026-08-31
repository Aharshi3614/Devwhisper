"""symbol_parser.py — Multi-language AST and regex symbol extraction for DevWhisper.

Supports extracting functions, methods, classes, structs, interfaces, and traits across
Python, JavaScript, TypeScript, Go, Rust, and Java source files with line numbers and signatures.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from language_detector import detect_language


@dataclass
class Symbol:
    """A logical code entity extracted from a source file."""

    name: str
    symbol_type: str  # "function", "class", "method", "struct", "interface", "trait", "enum"
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str]
    parent_class: Optional[str] = None
    language: Optional[str] = None
    signature: Optional[str] = None


# ---------------------------------------------------------------------------
# Python AST Extractor
# ---------------------------------------------------------------------------

class _PythonSymbolExtractor(ast.NodeVisitor):
    """Walk a Python AST and collect functions, classes, and methods."""

    def __init__(self, source_lines: List[str]) -> None:
        self.source_lines = source_lines
        self.symbols: List[Symbol] = []
        self._current_class: Optional[str] = None

    def _get_source(self, start: int, end: int) -> str:
        """Return the raw source text between *start* and *end* (1-based, inclusive)."""
        return "".join(self.source_lines[start - 1 : end])

    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Return the docstring body if the node has one, else None."""
        return ast.get_docstring(node)

    def _start_line(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
        """Return the first line of the node, accounting for decorators."""
        if node.decorator_list:
            return min(d.lineno for d in node.decorator_list)
        return getattr(node, "lineno", 1)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        start = self._start_line(node)
        end = getattr(node, "end_lineno", start)
        symbol_type = "method" if self._current_class else "function"

        # Build basic signature representation
        args_list = [a.arg for a in node.args.args]
        signature = f"def {node.name}({', '.join(args_list)})"

        symbol = Symbol(
            name=node.name,
            symbol_type=symbol_type,
            start_line=start,
            end_line=end,
            source=self._get_source(start, end),
            docstring=self._extract_docstring(node),
            parent_class=self._current_class,
            language="python",
            signature=signature,
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
            language="python",
            signature=f"class {node.name}",
        )
        self.symbols.append(symbol)

        old_class = self._current_class
        self._current_class = node.name
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self._current_class = old_class


# ---------------------------------------------------------------------------
# JavaScript & TypeScript Extractor
# ---------------------------------------------------------------------------

def _extract_js_ts_symbols(source: str) -> List[Symbol]:
    symbols: List[Symbol] = []
    lines = source.splitlines(keepends=True)
    if not lines:
        return symbols

    fn_pattern = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(")
    class_pattern = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z0-9_$]+)")
    arrow_pattern = re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_$]+)\s*=>"
    )
    interface_pattern = re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z0-9_$]+)")

    for idx, line in enumerate(lines):
        line_no = idx + 1
        m_fn = fn_pattern.match(line)
        if m_fn:
            name = m_fn.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="function",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="typescript",
                    signature=line.strip(),
                )
            )
            continue

        m_class = class_pattern.match(line)
        if m_class:
            name = m_class.group(1)
            end_line = min(len(lines), line_no + 40)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="class",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="typescript",
                    signature=line.strip(),
                )
            )
            continue

        m_arrow = arrow_pattern.match(line)
        if m_arrow:
            name = m_arrow.group(1)
            end_line = min(len(lines), line_no + 25)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="function",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="typescript",
                    signature=line.strip(),
                )
            )
            continue

        m_iface = interface_pattern.match(line)
        if m_iface:
            name = m_iface.group(1)
            end_line = min(len(lines), line_no + 25)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="interface",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="typescript",
                    signature=line.strip(),
                )
            )

    return symbols


# ---------------------------------------------------------------------------
# Go Extractor
# ---------------------------------------------------------------------------

def _extract_go_symbols(source: str) -> List[Symbol]:
    symbols: List[Symbol] = []
    lines = source.splitlines(keepends=True)
    if not lines:
        return symbols

    # Func: func Name(...) or func (r *Receiver) Name(...)
    method_pattern = re.compile(r"^\s*func\s+\(\s*[^)]+\s*\)\s*([A-Za-z0-9_]+)\s*\(")
    func_pattern = re.compile(r"^\s*func\s+([A-Za-z0-9_]+)\s*\(")
    struct_pattern = re.compile(r"^\s*type\s+([A-Za-z0-9_]+)\s+struct\b")
    interface_pattern = re.compile(r"^\s*type\s+([A-Za-z0-9_]+)\s+interface\b")

    for idx, line in enumerate(lines):
        line_no = idx + 1
        m_meth = method_pattern.match(line)
        if m_meth:
            name = m_meth.group(1)
            end_line = min(len(lines), line_no + 35)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="method",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="go",
                    signature=line.strip(),
                )
            )
            continue

        m_func = func_pattern.match(line)
        if m_func:
            name = m_func.group(1)
            end_line = min(len(lines), line_no + 35)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="function",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="go",
                    signature=line.strip(),
                )
            )
            continue

        m_struct = struct_pattern.match(line)
        if m_struct:
            name = m_struct.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="struct",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="go",
                    signature=line.strip(),
                )
            )
            continue

        m_iface = interface_pattern.match(line)
        if m_iface:
            name = m_iface.group(1)
            end_line = min(len(lines), line_no + 25)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="interface",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="go",
                    signature=line.strip(),
                )
            )

    return symbols


# ---------------------------------------------------------------------------
# Rust Extractor
# ---------------------------------------------------------------------------

def _extract_rust_symbols(source: str) -> List[Symbol]:
    symbols: List[Symbol] = []
    lines = source.splitlines(keepends=True)
    if not lines:
        return symbols

    fn_pattern = re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\(")
    struct_pattern = re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?struct\s+([A-Za-z0-9_]+)")
    enum_pattern = re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?enum\s+([A-Za-z0-9_]+)")
    trait_pattern = re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?trait\s+([A-Za-z0-9_]+)")
    impl_pattern = re.compile(r"^\s*impl(?:\s*<[^>]*>)?\s+(?:[A-Za-z0-9_:]+\s+for\s+)?([A-Za-z0-9_:]+)")

    for idx, line in enumerate(lines):
        line_no = idx + 1
        m_fn = fn_pattern.match(line)
        if m_fn:
            name = m_fn.group(1)
            end_line = min(len(lines), line_no + 35)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="function",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="rust",
                    signature=line.strip(),
                )
            )
            continue

        m_struct = struct_pattern.match(line)
        if m_struct:
            name = m_struct.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="struct",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="rust",
                    signature=line.strip(),
                )
            )
            continue

        m_enum = enum_pattern.match(line)
        if m_enum:
            name = m_enum.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="enum",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="rust",
                    signature=line.strip(),
                )
            )
            continue

        m_trait = trait_pattern.match(line)
        if m_trait:
            name = m_trait.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="trait",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="rust",
                    signature=line.strip(),
                )
            )

    return symbols


# ---------------------------------------------------------------------------
# Java Extractor
# ---------------------------------------------------------------------------

def _extract_java_symbols(source: str) -> List[Symbol]:
    symbols: List[Symbol] = []
    lines = source.splitlines(keepends=True)
    if not lines:
        return symbols

    class_pattern = re.compile(r"^\s*(?:(?:public|protected|private|abstract|final|static)\s+)*class\s+([A-Za-z0-9_]+)")
    interface_pattern = re.compile(r"^\s*(?:(?:public|protected|private|static)\s+)*interface\s+([A-Za-z0-9_]+)")
    enum_pattern = re.compile(r"^\s*(?:(?:public|protected|private|static)\s+)*enum\s+([A-Za-z0-9_]+)")
    method_pattern = re.compile(
        r"^\s*(?:(?:public|protected|private|static|final|synchronized|abstract)\s+)*(?:[\w<>\[\]]+\s+)+([A-Za-z0-9_]+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{"
    )


    for idx, line in enumerate(lines):
        line_no = idx + 1
        m_class = class_pattern.match(line)
        if m_class:
            name = m_class.group(1)
            end_line = min(len(lines), line_no + 45)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="class",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="java",
                    signature=line.strip(),
                )
            )
            continue

        m_iface = interface_pattern.match(line)
        if m_iface:
            name = m_iface.group(1)
            end_line = min(len(lines), line_no + 35)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="interface",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="java",
                    signature=line.strip(),
                )
            )
            continue

        m_enum = enum_pattern.match(line)
        if m_enum:
            name = m_enum.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="enum",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="java",
                    signature=line.strip(),
                )
            )
            continue

        m_method = method_pattern.match(line)
        if m_method:
            name = m_method.group(1)
            end_line = min(len(lines), line_no + 30)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="method",
                    start_line=line_no,
                    end_line=end_line,
                    source="".join(lines[line_no - 1 : end_line]),
                    docstring=None,
                    language="java",
                    signature=line.strip(),
                )
            )


    return symbols


# ---------------------------------------------------------------------------
# Unified Public API
# ---------------------------------------------------------------------------

def extract_symbols_from_source(source: str, filename: str = "<unknown>") -> List[Symbol]:
    """Parse *source* and return all extractable symbols across supported languages.

    Args:
        source: Full source code.
        filename: Used for format detection and error reporting.

    Returns:
        A list of :class:`Symbol` objects.
    """
    lang = detect_language(filename)
    if lang in ("javascript", "typescript"):
        return _extract_js_ts_symbols(source)
    elif lang == "go":
        return _extract_go_symbols(source)
    elif lang == "rust":
        return _extract_rust_symbols(source)
    elif lang == "java":
        return _extract_java_symbols(source)

    # Fallback / default: Python AST parser
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [""]

    extractor = _PythonSymbolExtractor(lines)
    extractor.visit(tree)
    return extractor.symbols


def extract_symbols_from_file(filepath: str) -> List[Symbol]:
    """Read *filepath* and return all extractable symbols.

    Args:
        filepath: Path to a supported source file.

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
