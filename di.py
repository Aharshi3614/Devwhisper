"""di.py — Simple dependency container for test-friendly overrides.

Provides lazy factories and setter hooks so tests and integration code can
replace production components (e.g., SessionManager, indexing queue) with
mocks or alternatives without editing application code.

This is intentionally small and non-opinionated — it does not implement a full
IoC container, just enough indirection to make components overridable in tests.
"""

from __future__ import annotations

import queue as _queue
from typing import Any

try:
    # Import concrete classes used by default factories
    from session_manager import SessionManager
    import repositories as _repositories
except Exception:  # pragma: no cover - defensive when imported in tests that stub modules
    SessionManager = None
    _repositories = None

# Module-level singletons (lazy-initialized)
_session_manager: SessionManager | None = None
_indexing_queue: _queue.Queue | None = None
_jobs_history: list | None = None
_repositories_override: Any = None


def get_session_manager(max_sessions: int = 100, max_history_per_session: int = 5) -> SessionManager:
    """Return a lazily-created SessionManager. Tests can call set_session_manager() to override."""
    global _session_manager
    if _session_manager is None:
        if SessionManager is None:
            raise RuntimeError("SessionManager class is unavailable in this environment")
        _session_manager = SessionManager(max_sessions=max_sessions, max_history_per_session=max_history_per_session)
    return _session_manager


def set_session_manager(obj: SessionManager) -> None:
    """Override the session manager (useful for tests)."""
    global _session_manager
    _session_manager = obj


def get_indexing_queue() -> _queue.Queue:
    """Return a lazily-created indexing queue. Tests can override with set_indexing_queue()."""
    global _indexing_queue
    if _indexing_queue is None:
        _indexing_queue = _queue.Queue()
    return _indexing_queue


def set_indexing_queue(q: _queue.Queue) -> None:
    global _indexing_queue
    _indexing_queue = q


def get_jobs_history() -> list:
    """Return a singleton list used to record indexing job metadata."""
    global _jobs_history
    if _jobs_history is None:
        _jobs_history = []
    return _jobs_history


def set_jobs_history(l: list) -> None:
    global _jobs_history
    _jobs_history = l


def get_repositories():
    """Return repositories module or test-provided override."""
    return _repositories_override or _repositories


def set_repositories(obj: Any) -> None:
    """Override the repositories module used by the application (for tests)."""
    global _repositories_override
    _repositories_override = obj


__all__ = [
    "get_session_manager",
    "set_session_manager",
    "get_indexing_queue",
    "set_indexing_queue",
    "get_jobs_history",
    "set_jobs_history",
    "get_repositories",
    "set_repositories",
]
