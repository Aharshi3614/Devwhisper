"""language_detector.py — Programming language detection and source metadata utility.

This module inspects file extensions, filenames, and source contents to reliably detect
the programming language, comment styles, and structural properties for AST and symbol parsing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class LanguageSpec:
    """Specification of a programming language supported by DevWhisper."""

    name: str
    extensions: Set[str]
    line_comment_prefixes: tuple[str, ...]
    block_comment_delimiters: Optional[tuple[str, str]]
    is_code: bool = True


# Registry of known languages
LANGUAGE_REGISTRY: Dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        extensions={".py", ".pyw", ".pyi"},
        line_comment_prefixes=("#",),
        block_comment_delimiters=('"""', '"""'),
    ),
    "javascript": LanguageSpec(
        name="javascript",
        extensions={".js", ".jsx", ".mjs", ".cjs"},
        line_comment_prefixes=("//",),
        block_comment_delimiters=("/*", "*/"),
    ),
    "typescript": LanguageSpec(
        name="typescript",
        extensions={".ts", ".tsx", ".mts", ".cts"},
        line_comment_prefixes=("//",),
        block_comment_delimiters=("/*", "*/"),
    ),
    "go": LanguageSpec(
        name="go",
        extensions={".go"},
        line_comment_prefixes=("//",),
        block_comment_delimiters=("/*", "*/"),
    ),
    "rust": LanguageSpec(
        name="rust",
        extensions={".rs"},
        line_comment_prefixes=("//",),
        block_comment_delimiters=("/*", "*/"),
    ),
    "java": LanguageSpec(
        name="java",
        extensions={".java"},
        line_comment_prefixes=("//",),
        block_comment_delimiters=("/*", "*/"),
    ),
    "c_cpp": LanguageSpec(
        name="c_cpp",
        extensions={".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"},
        line_comment_prefixes=("//",),
        block_comment_delimiters=("/*", "*/"),
    ),
    "markdown": LanguageSpec(
        name="markdown",
        extensions={".md", ".markdown"},
        line_comment_prefixes=(),
        block_comment_delimiters=("<!--", "-->"),
        is_code=False,
    ),
    "json": LanguageSpec(
        name="json",
        extensions={".json"},
        line_comment_prefixes=(),
        block_comment_delimiters=None,
        is_code=False,
    ),
    "yaml": LanguageSpec(
        name="yaml",
        extensions={".yaml", ".yml"},
        line_comment_prefixes=("#",),
        block_comment_delimiters=None,
        is_code=False,
    ),
}

_EXT_TO_LANG: Dict[str, str] = {}
for lang_name, spec in LANGUAGE_REGISTRY.items():
    for ext in spec.extensions:
        _EXT_TO_LANG[ext] = lang_name


def detect_language(filepath: str | Path) -> Optional[str]:
    """Detect the language of a file based on its extension or filename.

    Args:
        filepath: Path or filename to inspect.

    Returns:
        The canonical language name (e.g., "python", "go", "rust") or None if unrecognized.
    """
    ext = Path(filepath).suffix.lower()
    return _EXT_TO_LANG.get(ext)


def get_language_spec(language_name: str) -> Optional[LanguageSpec]:
    """Retrieve the LanguageSpec for a given language name."""
    return LANGUAGE_REGISTRY.get(language_name.lower())


def is_code_file(filepath: str | Path) -> bool:
    """Check if the given file is considered a source code file."""
    lang = detect_language(filepath)
    if not lang:
        return False
    spec = get_language_spec(lang)
    return bool(spec and spec.is_code)
