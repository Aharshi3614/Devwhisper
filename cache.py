"""
cache.py — In-memory LRU cache for repeated query responses with similarity matching.

This module provides a thread-safe bounded cache to avoid re-running the
retrieval + LLM pipeline for identical or near-duplicate queries. Only successful,
non-empty responses are cached.

Design:
    - Uses OrderedDict for O(1) LRU operations (get, move_to_end, popitem).
    - Thread-safe via threading.Lock.
    - Entries are scoped to the repository that produced them (issue #258).
    - Near-duplicate detection via Jaccard similarity over tokenized queries.
    - Configurable similarity threshold (default 0.70 catches common word variations).

Repository scoping:
    DevWhisper supports several repositories, each with its own Qdrant
    collection, BM25 index and index cache. The response cache sits in front
    of all of that, so it has to respect the same isolation — otherwise an
    answer generated for repository A is served for repository B after a
    ``POST /repos/switch``, complete with a "Sources used:" footer naming
    files that are not in the active repository.

    The scope is read from ``repositories.get_current_repo_id()`` inside this
    module rather than passed in by callers, so no call site can forget it.

Public API:
    get(query, threshold) → cached response or None
    put(query, response)  → store response (skips empty responses)
    invalidate_repo(repo_id) → drop one repository's entries (e.g. after re-index)
    clear() → drop everything
"""

import threading
from collections import OrderedDict
from typing import Set, Tuple

try:
    from config import CACHE_SIMILARITY_THRESHOLD
except ImportError:
    # Default fallback threshold if config import fails (0.70 catches common word variations)
    CACHE_SIMILARITY_THRESHOLD = 0.70

try:
    import repositories
except ImportError:  # pragma: no cover - defensive, keeps the cache usable standalone
    repositories = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_CACHE_SIZE = 50
"""Maximum number of entries in the LRU cache, across all repositories.

Deliberately a global bound rather than a per-repository one: the memory
ceiling should not grow with the number of registered repositories.
"""

# ---------------------------------------------------------------------------
# Internal storage (protected by the lock below)
# ---------------------------------------------------------------------------
# Keys are (repo_id, normalized_query) pairs. repo_id is None when no
# repository is registered, which keeps single-repo behaviour unchanged.
# OrderedDict preserves insertion order and gives O(1) move/pop operations,
# making it the canonical Python LRU cache implementation.
CacheKey = Tuple[str | None, str]
_cache: "OrderedDict[CacheKey, str]" = OrderedDict()

# Guards all mutations and reads of _cache.
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _current_scope() -> str | None:
    """
    Return the id of the repository whose answers we are caching.

    Returns:
        The active repository id, or None when no repository is registered
        (or the registry is unavailable). None is a scope like any other, so
        single-repository deployments behave exactly as before.
    """
    if repositories is None:
        return None
    try:
        return repositories.get_current_repo_id()
    except Exception:
        # A broken registry must not take down answering — fall back to the
        # unscoped bucket rather than raising out of a cache lookup.
        return None


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

    Only entries cached for the currently active repository are considered —
    both for the exact match and for the near-duplicate scan.

    Lookup strategy:
        1. Exact normalized key match within the active repository (O(1)).
        2. If missing, perform similarity check against the active
           repository's cached keys. If similarity >= threshold, reuse the
           cached response.

    On a hit, the matched entry is promoted to most-recently-used.

    Args:
        query: User's query string.
        threshold: Minimum Jaccard similarity for near-duplicate matching.

    Returns:
        Cached response string, or None if no match found.
    """
    scope = _current_scope()
    key = _normalize(query)

    with _cache_lock:
        # 1. Exact Match Check
        value = _cache.get((scope, key))
        if value is not None:
            _cache.move_to_end((scope, key))
            return value

        # 2. Near-Duplicate Similarity Check
        query_tokens = _tokenize(key)
        if not query_tokens:
            return None

        best_match_key = None
        best_score = 0.0

        for cached_key in reversed(_cache):  # Check most recent entries first
            cached_scope, cached_query = cached_key
            if cached_scope != scope:
                # Belongs to a different repository — not a candidate.
                continue

            cached_tokens = _tokenize(cached_query)
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

    The entry is filed under the repository that is active right now, so it
    can only ever be read back while that repository is active.

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

    key = (_current_scope(), _normalize(query))
    with _cache_lock:
        _cache[key] = response
        _cache.move_to_end(key)

        # Evict the least-recently-used entry if we are over capacity.
        while len(_cache) > MAX_CACHE_SIZE:
            _cache.popitem(last=False)


def invalidate_repo(repo_id: str | None) -> int:
    """
    Drop every cached answer belonging to *repo_id*.

    Useful after re-indexing a repository: the code changed, so answers
    generated from the previous snapshot are stale.

    Args:
        repo_id: Repository id whose entries should be removed. ``None``
            targets the unscoped bucket used when no repository is registered.

    Returns:
        Number of entries removed.
    """
    with _cache_lock:
        doomed = [key for key in _cache if key[0] == repo_id]
        for key in doomed:
            del _cache[key]
        return len(doomed)


def clear() -> int:
    """
    Drop every cached answer, for all repositories.

    Returns:
        Number of entries removed.
    """
    with _cache_lock:
        removed = len(_cache)
        _cache.clear()
        return removed


def size() -> int:
    """Return the total number of cached entries across all repositories."""
    with _cache_lock:
        return len(_cache)


def get_stats() -> dict:
    """Return cache statistics for answers and retrieval cache."""
    try:
        from retrieval_cache import retrieval_cache
        retrieval_stats = retrieval_cache.get_stats()
    except Exception:
        retrieval_stats = {}

    with _cache_lock:
        return {
            "answer_cache": {
                "size": len(_cache),
                "max_size": MAX_CACHE_SIZE,
            },
            "retrieval_cache": retrieval_stats,
        }

