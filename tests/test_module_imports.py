"""Import smoke tests for every top-level module (issue #255).

`retriever.py` shipped with `RequestContext` used in the `retrieve()` signature
but never imported. Because the module has no `from __future__ import
annotations`, Python evaluates that annotation while defining the function, so
the module raised `NameError` on import. That took down the FastAPI app and
made 18 test modules fail during collection — a wall of identical errors that
says nothing about the actual cause.

These tests are the cheap guard for that class of failure:

  1. Every top-level module still imports.
  2. Every annotation in those modules resolves to a real name.

(2) matters because a lazily-evaluated annotation — under `from __future__
import annotations`, or a string annotation — hides the same mistake until
something calls `typing.get_type_hints()` at runtime. Resolving the hints here
surfaces it at test time instead.
"""

import importlib
import inspect
import os
import pkgutil
import typing

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules deliberately left out of the sweep:
#   download_model — downloads the embedding model at import time.
#   conftest       — pytest bootstrap, already imported by the runner.
EXCLUDED_MODULES = {"download_model", "conftest"}


def _top_level_modules() -> list[str]:
    """Return the importable top-level module names in the repository root."""
    names = [
        name
        for _finder, name, ispkg in pkgutil.iter_modules([REPO_ROOT])
        if not ispkg and not name.startswith("_") and name not in EXCLUDED_MODULES
    ]
    return sorted(names)


def _annotated_objects(module):
    """Yield (label, callable) pairs for functions/methods defined in *module*."""
    for obj_name, obj in vars(module).items():
        if obj_name.startswith("_"):
            continue

        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            yield f"{module.__name__}.{obj_name}", obj

        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for attr_name, attr in vars(obj).items():
                if inspect.isfunction(attr):
                    yield f"{module.__name__}.{obj_name}.{attr_name}", attr


def test_top_level_modules_are_discovered():
    """Guard the discovery itself — an empty sweep would pass vacuously."""
    modules = _top_level_modules()

    assert "retriever" in modules
    assert "main" in modules
    assert "indexer" in modules
    assert "download_model" not in modules
    assert len(modules) >= 15


@pytest.mark.parametrize("module_name", _top_level_modules())
def test_module_imports_cleanly(module_name):
    """Each top-level module imports without raising.

    This is the direct regression test for #255: `import retriever` raised
    `NameError: name 'RequestContext' is not defined`.
    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - only runs when broken
        pytest.fail(
            f"`import {module_name}` failed with "
            f"{type(exc).__name__}: {exc}\n"
            "A top-level module must always be importable — a failure here "
            "breaks the app at startup and every test module that imports it."
        )


@pytest.mark.parametrize("module_name", _top_level_modules())
def test_module_annotations_resolve(module_name):
    """Every annotation in the module refers to a name the module can see.

    Catches the #255 mistake even when the annotation is evaluated lazily,
    which is where it would otherwise stay hidden until runtime.
    """
    module = importlib.import_module(module_name)

    unresolved = []
    for label, func in _annotated_objects(module):
        if not getattr(func, "__annotations__", None):
            continue
        try:
            typing.get_type_hints(func)
        except NameError as exc:
            unresolved.append(f"{label}: {exc}")
        except Exception:
            # Anything other than an undefined name (e.g. an exotic generic
            # this Python version cannot introspect) is out of scope here.
            continue

    assert not unresolved, (
        "Unresolvable type annotations found:\n  " + "\n  ".join(unresolved)
    )


def test_retriever_exposes_request_context():
    """`retrieve()` advertises RequestContext, so the name must be bound."""
    import retriever

    assert hasattr(retriever, "RequestContext"), (
        "retriever.retrieve() annotates its `query` and `context` parameters "
        "with RequestContext, so the module has to import it."
    )

    hints = typing.get_type_hints(retriever.retrieve)
    assert "query" in hints
    assert "context" in hints


def test_retriever_import_does_not_depend_on_import_order():
    """Re-importing retriever from a clean module state still works.

    A stale `sys.modules` entry can mask a missing import when the name
    happens to be present from an earlier import elsewhere. Forcing a reload
    proves `retriever` supplies its own dependency.
    """
    import sys

    import retriever

    saved = sys.modules.pop("retriever")
    try:
        reloaded = importlib.import_module("retriever")
        assert hasattr(reloaded, "RequestContext")
        assert callable(reloaded.retrieve)
    finally:
        sys.modules["retriever"] = saved
