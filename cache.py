"""In-memory LRU cache for repeated query responses with similarity matching.

This module provides a thread-safe bounded cache to avoid re-running the
retrieval + LLM pipeline for identical or near-duplicate queries. Only successful,
non-empty responses are cached.
"""

import threading
from collections import OrderedDict
from typing import Set

try:
    from config import CACHE_SIMILARITY_THRESHOLD
except ImportError:
    # Default fallback threshold if config import fails (0.70 catches common word variations)
    CACHE_SIMILARITY_THRESHOLD = 0.70


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


def _tokenize(text: str) -> Set[str]:
    """Tokenize normalized query string into a set of words."""
    return set(_normalize(text).split())


def _calculate_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Calculate Jaccard similarity score between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get(query: str, threshold: float = CACHE_SIMILARITY_THRESHOLD) -> str | None:
    """Return a cached response for *query*, or ``None`` on a cache miss.

    1. Checks for an exact normalized key match first (O(1)).
    2. If missing, performs a similarity check against active cache keys.
       If similarity >= threshold, reuses the cached response.
    
    On a hit, the matched entry is promoted to most-recently-used.
    """
    key = _normalize(query)
    
    with _cache_lock:
        # 1. Exact Match Check
        value = _cache.get(key)
        if value is not None:
            _cache.move_to_end(key)
            return value

        # 2. Near-Duplicate Similarity Check
        query_tokens = _tokenize(key)
        if not query_tokens:
            return None

        best_match_key = None
        best_score = 0.0

        for cached_key in reversed(_cache):  # Check most recent entries first
            cached_tokens = _tokenize(cached_key)
            score = _calculate_similarity(query_tokens, cached_tokens)

            if score > best_score:
                best_score = score
                best_match_key = cached_key

            # Early exit if we find a very high match
            if best_score >= 0.95:
                break

        if best_match_key is not None and best_score >= threshold:
            # Promote matched key and return cached response
            _cache.move_to_end(best_match_key)
            return _cache[best_match_key]

        return None


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
        while len(_cache) > MAX_CACHE_SIZE:
            _cache.popitem(last=False)
            