"""In-memory LRU cache for repeated query responses.

This module provides a thread-safe bounded cache to avoid re-running the
retrieval + LLM pipeline for identical queries. Only successful, non-empty
responses are cached.
"""

import threading
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_CACHE_SIZE = 50


# ---------------------------------------------------------------------------
# Internal storage (protected by the lock below)
# ---------------------------------------------------------------------------
# OrderedDict preserves insertion order and gives O(1) move/pop operations,
# making it the canonical Python LRU cache implementation.
_cache: OrderedDict[str, str] = OrderedDict()

# Guards all mutations and reads of _cache.
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(query: str) -> str:
    """Normalize a query string to a stable cache key.

    - Strips leading/trailing whitespace
    - Converts to lowercase
    - Collapses consecutive whitespace into a single space
    """
    return " ".join(query.strip().lower().split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get(query: str) -> str | None:
    """Return a cached response for *query*, or ``None`` on a cache miss.

    On a hit the entry is promoted to most-recently-used.
    """
    key = _normalize(query)
    with _cache_lock:
        value = _cache.get(key)
        if value is not None:
            # Promote to most-recently-used.
            _cache.move_to_end(key)
        return value


def put(query: str, response: str) -> None:
    """Store a successful *response* for *query* in the cache.

    Empty responses and failures should not be passed here; the caller is
    responsible for only caching successful, non-empty answers.
    """
    if not response or not response.strip():
        return

    key = _normalize(query)
    with _cache_lock:
        _cache[key] = response
        _cache.move_to_end(key)

        # Evict the least-recently-used entry if we are over capacity.
        # popitem(last=False) removes the oldest item in O(1).
        while len(_cache) > MAX_CACHE_SIZE:
            _cache.popitem(last=False)
