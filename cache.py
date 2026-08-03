"""
cache.py — In-memory LRU cache for repeated query responses with similarity matching.

This module provides a thread-safe bounded cache to avoid re-running the
retrieval + LLM pipeline for identical or near-duplicate queries. Only successful,
non-empty responses are cached.

Design:
    - Uses OrderedDict for O(1) LRU operations (get, move_to_end, popitem).
    - Thread-safe via threading.Lock.
    - Near-duplicate detection via Jaccard similarity over tokenized queries.
    - Configurable similarity threshold (default 0.70 catches common word variations).

Public API:
    get(query, threshold) → cached response or None
    put(query, response)  → store response (skips empty responses)
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
"""Maximum number of entries in the LRU cache."""

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
    """
    Normalize a query string to a stable cache key.

    Steps:
        1. Strip leading/trailing whitespace.
        2. Convert to lowercase.
        3. Collapse consecutive whitespace into a single space.

    Args:
        query: Raw query string.

    Returns:
        Normalized, stable cache key string.
    """
    return " ".join(query.strip().lower().split())


def _tokenize(text: str) -> Set[str]:
    """
    Tokenize a normalized query string into a set of unique words.

    Args:
        text: Normalized query string.

    Returns:
        Set of word tokens.
    """
    return set(_normalize(text).split())


def _calculate_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """
    Calculate Jaccard similarity score between two token sets.

    Jaccard similarity = |intersection| / |union|

    Args:
        tokens1: First set of word tokens.
        tokens2: Second set of word tokens.

    Returns:
        Similarity score in range [0.0, 1.0].
    """
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get(query: str, threshold: float = CACHE_SIMILARITY_THRESHOLD) -> str | None:
    """
    Return a cached response for *query*, or ``None`` on a cache miss.

    Lookup strategy:
        1. Exact normalized key match (O(1)).
        2. If missing, perform similarity check against active cache keys.
           If similarity >= threshold, reuse the cached response.

    On a hit, the matched entry is promoted to most-recently-used.

    Args:
        query: User's query string.
        threshold: Minimum Jaccard similarity for near-duplicate matching.

    Returns:
        Cached response string, or None if no match found.
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
    """
    Store a successful *response* for *query* in the cache.

    Empty responses and failures should not be passed here; the caller is
    responsible for only caching successful, non-empty answers.

    Args:
        query: User's query string.
        response: The assistant's response to cache.

    Side effects:
        Adds or updates the cache entry. Evicts the least-recently-used entry
        if capacity is exceeded.
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
            
