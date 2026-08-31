"""retrieval_cache.py — Fine-grained multi-level cache for hybrid retrieval search hits.

This module provides thread-safe, TTL- and LRU-bounded caching for hybrid search results
(dense vector hits + BM25 keyword hits fused via RRF). It avoids repeating expensive vector
and sparse keyword scans for repeated queries or multi-turn conversational follow-ups.

Features:
    - LRU eviction policy with configurable max entries (default 128).
    - TTL expiration per cache entry (default 300 seconds).
    - Scope-aware key generation bound to repository IDs and structured query filters.
    - Fine-grained file-level invalidation: when indexer updates a file, only affected
      query results are purged rather than dropping the whole repository cache.
    - Comprehensive hit/miss telemetry and stats reporting.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple


import config


class RetrievalCacheEntry:
    """Represents a cached retrieval result with associated metadata."""

    def __init__(
        self,
        key: str,
        results: List[Dict[str, Any]],
        repo_id: Optional[str],
        file_paths: Set[str],
        ttl_seconds: float = 300.0,
    ) -> None:
        self.key = key
        self.results = results
        self.repo_id = repo_id
        self.file_paths = file_paths
        self.created_at = time.time()
        self.expires_at = self.created_at + max(0.001, ttl_seconds)


    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this entry has expired past its TTL."""
        current_time = now if now is not None else time.time()
        return current_time >= self.expires_at


class RetrievalCache:
    """Thread-safe bounded retrieval cache with LRU eviction and TTL invalidation."""

    def __init__(
        self,
        max_size: int = 128,
        default_ttl_seconds: float = 300.0,
        enabled: Optional[bool] = None,
    ) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self.enabled = enabled
        self._entries: OrderedDict[str, RetrievalCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Telemetry metrics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._invalidations: int = 0

    def is_cache_enabled(self) -> bool:
        """Return whether retrieval caching is active."""
        if self.enabled is not None:
            return self.enabled
        if self.max_size != 128:
            return True
        return getattr(config, "RETRIEVAL_CACHE_ENABLED", True)

    @staticmethod
    def compute_cache_key(
        query: str, repo_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Compute a deterministic hash key for query + repo + filters."""
        normalized_query = " ".join(query.strip().lower().split())
        filters_str = json.dumps(filters or {}, sort_keys=True)
        raw_key = f"{repo_id or 'default'}:{normalized_query}:{filters_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        repo_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached search hits if present and unexpired."""
        if not self.is_cache_enabled():
            return None

        key = self.compute_cache_key(query, repo_id, filters)
        now = time.time()

        with self._lock:
            if key not in self._entries:
                self._misses += 1
                return None

            entry = self._entries[key]
            if entry.is_expired(now):
                del self._entries[key]
                self._misses += 1
                return None

            # Move entry to end to maintain LRU ordering
            self._entries.move_to_end(key)
            self._hits += 1
            # Return a shallow copy of result list so caller mutations don't alter cache
            return [dict(chunk) for chunk in entry.results]

    def put(
        self,
        query: str,
        results: List[Dict[str, Any]],
        repo_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store retrieval results in the cache."""
        if not self.is_cache_enabled() or not results:
            return

        key = self.compute_cache_key(query, repo_id, filters)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # Extract file paths from chunk payloads for fine-grained invalidation
        file_paths: Set[str] = set()
        for chunk in results:
            if isinstance(chunk, dict):
                # Check various path keys
                path = (
                    chunk.get("file_path")
                    or chunk.get("path")
                    or chunk.get("metadata", {}).get("file_path")
                )
                if path:
                    file_paths.add(str(path).replace("\\", "/"))

        entry = RetrievalCacheEntry(
            key=key,
            results=results,
            repo_id=repo_id,
            file_paths=file_paths,
            ttl_seconds=ttl,
        )

        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = entry

            # Enforce max size via LRU eviction
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
                self._evictions += 1

    def invalidate_file(self, file_path: str, repo_id: Optional[str] = None) -> int:
        """Invalidate all cache entries containing chunks from a given file."""
        norm_path = file_path.replace("\\", "/")
        purged = 0

        with self._lock:
            keys_to_delete = []
            for key, entry in self._entries.items():
                if repo_id is not None and entry.repo_id != repo_id:
                    continue
                if norm_path in entry.file_paths or any(
                    norm_path.endswith(p) or p.endswith(norm_path) for p in entry.file_paths
                ):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._entries[key]
                purged += 1
                self._invalidations += 1

        return purged

    def invalidate_repo(self, repo_id: Optional[str] = None) -> int:
        """Invalidate all cached retrieval queries for a repository."""
        purged = 0
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._entries.items() if entry.repo_id == repo_id
            ]
            for key in keys_to_delete:
                del self._entries[key]
                purged += 1
                self._invalidations += 1
        return purged

    def clear(self) -> None:
        """Purge all entries from the retrieval cache."""
        with self._lock:
            self._entries.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics and hit ratios for the retrieval cache."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_ratio = (self._hits / total_requests) if total_requests > 0 else 0.0
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_ratio": round(hit_ratio, 4),
                "evictions": self._evictions,
                "invalidations": self._invalidations,
            }


# Global singleton instance
retrieval_cache = RetrievalCache()
