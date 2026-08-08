"""circular_import_checker.py — Detect circular imports in indexed Python files.

A circular import happens when module A imports module B, and B (directly or
indirectly) imports A back. At runtime this can raise ``ImportError``. This
module parses ``import`` statements in the files DevWhisper indexes and
reports any circular chains it finds, so users can fix them before they
break the app.

The checks are read-only and have no effect on the indexing pipeline, so
existing indexing functionality stays unaffected.

Key functions:
  - parse_import(file_path, file_name): return {file_name: [modules it imports]}.
  - tokenize_lines_identifiers(python_code): return the module names in code.
  - check_import_circular(files_depend): find circular imports in a dependency map.
"""

import ast
import os

def parse_import(file_path: str, file_name: str) -> dict[str, list[str]]:
    """Parse a Python file's import statements.

    Args:
        file_path: Absolute path to the Python file to analyze.
        file_name: the name of the Python file to analyze.

    Returns:
        A dict mapping the file name to the list of module names it imports,
        e.g. ``{"a": ["os", "re"]}``. Empty list if the file imports
        nothing or cannot be read.
    """
    file_depend = {}
    if ".py" == os.path.splitext(file_name)[1]:
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()
            depended = tokenize_lines_identifiers(file_contents)
            file_depend[os.path.splitext(file_name)[0]] = depended

    return file_depend

def tokenize_lines_identifiers(python_code: str) -> list[str]:
    """Use AST to tokenize the lines identifier statements.
    Args:
        python_code: python code to tokenize.

    Returns:
        A list of imported module names.
        e.g. for this code:

            import os
            from sys import path
            import numpy as np
            print("hello world")

        it returns:

            ["os", "sys", "numpy"]
        """
    try:
        tree = ast.parse(python_code)
    except SyntaxError:
        return []

    result = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module is None: continue
            module = node.module if node.module is not None else ''
            result.append(module)

    return result

def check_import_circular(files_depend: dict[str, list[str]]) -> list[str]:
    """Find circular imports in a dependency map.

    A cycle exists when file A imports file B, and B (directly or indirectly)
    imports A back. Both direct two-file cycles and longer indirect chains
    (e.g. ``a -> b -> c -> a``) are detected.

    Args:
        files_depend: A dict mapping each file name to the list of module
            names it imports, e.g. ``{"a": ["b"], "b": ["c"], "c": ["a"]}``.

    Returns:
        A list of human-readable messages describing each file that is part
        of a cycle, e.g. ``["a is in a circular import"]``. Empty list if
        there are no circular imports.
    """
    import_circular = []

    for file_name in files_depend:
        if _has_cycle(file_name, [file_name], files_depend):
            import_circular.append(file_name+"is in a circular import")

    return import_circular

def _has_cycle(current: str, path: list[str], files_depend: dict[str, list[str]]) -> bool:
    """Walk the dependency graph from ``current`` and report whether a cycle is found.

    This is a depth-first walk: starting at ``current``, it follows each
    import (``imported_name``) in turn. If an import leads back to a module
    already recorded in ``path``, a cycle exists and ``True`` is returned.
    Otherwise the walk continues deeper into that module (recursively).

    Args:
        current: The module name to start the walk from.
        path: The list of module names already visited on the current route,
            e.g. ``["a", "b"]`` means we walked from a to b. Starts as
            ``[start_module]``.
        files_depend: The dependency map: each module name to the list of
            modules it imports.

    Returns:
        True if walking from ``current`` reaches a module already in ``path``
        (a cycle); False if every route ends at a non-indexed module.
    """
    for imported_name in files_depend.get(current, []):
        if imported_name in path:
            return True
        if imported_name in files_depend:
            if _has_cycle(imported_name, path+[imported_name], files_depend):
                return True
    return False