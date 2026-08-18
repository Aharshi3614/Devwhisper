"""retriever.py — Hybrid retrieval engine for DevWhisper.

This module implements the core search functionality that powers DevWhisper's
codebase Q&A. It combines three retrieval strategies via Reciprocal Rank Fusion (RRF):

  1. Dense vector search (Qdrant + sentence-transformers embeddings)
  2. Sparse keyword search (BM25 over tokenized code chunks)
  3. Exact symbol matching (function/class name extraction from queries)

The fused results are formatted into a structured context string suitable for
LLM consumption, including file paths, symbol names, line numbers, and docstrings.

Key components:
  - retrieve(): Main entry point — hybrid search + context formatting across single or multiple repositories.
  - _fuse_ranked_lists(): Picks between fusion and a single-retriever result.
  - preprocess_query(): Normalize user queries before embedding/search.
  - _keyword_search(): BM25-based keyword retrieval.
  - _extract_symbols(): Heuristic symbol extraction from natural language queries.
  - _exact_symbol_search(): Direct symbol name matching in chunks (metadata-aware).
  - _rrf_fusion(): Reciprocal Rank Fusion to combine ranked lists.
  - _fusion_key(): Content identity used to recognise the same chunk across lists.
  - check_embedding_version(): Warns if indexed embeddings differ from config.
  - get_repository_metadata(): Reads indexing metadata for a repository's index cache.
  - metadata_path_for(): Resolves a repository id to its index-cache path.

Dependencies:
  - QdrantClient (vector DB)
  - SentenceTransformer (dense embeddings)
  - rank_bm25 (sparse keyword search)
"""

import json
import os
import pickle
import re

from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

import repositories as repo_registry
import vector_store

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    QDRANT_COLLECTION_NAME,
    QDRANT_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
    BM25_INDEX_PATH,
    HYBRID_TOP_K,
    RRF_K,
)
from logger import logger
from request_context import RequestContext

# ---------------------------------------------------------------------------
# Qdrant client and embedder (module-level singletons)
# ---------------------------------------------------------------------------
client = vector_store.client
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

# ---------------------------------------------------------------------------
# BM25 index (lazy-loaded per repository, reloaded when the file changes)
# ---------------------------------------------------------------------------
# Cached payload per repository: {repo_id: (stamp, data)} where `stamp` is the
# (mtime_ns, size) of the pickle the payload was loaded from, or None when the
# file was missing/unreadable.
#
# Re-indexing rewrites this pickle in the same process that holds the cached
# copy, so keying purely on repo_id — as this did originally — pinned keyword
# and symbol search to whatever snapshot happened to load first. Dense vector
# search queries Qdrant live and does see the new code, so the two halves of
# hybrid retrieval drifted apart after every re-index, and RRF still boosted
# the stale chunks because they showed up in two of the three ranked lists.
_bm25_data: dict[str | None, tuple[tuple[int, int] | None, dict | None]] = {}


def _file_stamp(path: str) -> tuple[int, int] | None:
    """
    Return a cheap change-detection stamp for the file at *path*.

    Uses ``(st_mtime_ns, st_size)`` — one stat() call, no reading and no
    hashing, so this is affordable on every query. Size is included because
    a rewrite within the same mtime tick is possible on coarse clocks.

    Args:
        path: Path to stat.

    Returns:
        The stamp tuple, or None if the file does not exist or cannot be stat'ed.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _bm25_stamp(path: str) -> tuple[int, int] | None:
    """Change-detection stamp for the BM25 pickle at *path*."""
    return _file_stamp(path)


def _bm25_index_path(repo_id: str | None) -> str:
    """Return the BM25 pickle path for *repo_id* (or the shared default)."""
    if repo_id is None:
        return BM25_INDEX_PATH
    return repo_registry.bm25_path(repo_id)


def invalidate_bm25_cache(repo_id: str | None = None) -> None:
    """
    Drop the in-process BM25 cache so the next query reloads from disk.

    ``_get_bm25()`` already notices a changed file on its own; this is for
    callers that know the index just changed and would rather not wait for
    the stat() to say so.

    Args:
        repo_id: Repository whose cached index to drop. ``None`` clears the
            entry for the shared/default index only — pass no argument and
            use :func:`clear_bm25_cache` to drop everything.
    """
    _bm25_data.pop(repo_id, None)


def clear_bm25_cache() -> None:
    """Drop every cached BM25 index (all repositories)."""
    _bm25_data.clear()


def _get_bm25(repo_id: str | None) -> dict | None:
    """
    Return the BM25 payload for *repo_id*, reloading it if the file changed.

    The pickle is re-read only when its (mtime, size) stamp differs from the
    one the cached payload was loaded with, so a steady-state query pays a
    single stat() rather than a pickle.load().

    A missing or unreadable index is cached as ``None`` along with its stamp,
    which means a repository that had no BM25 index when it was first queried
    starts working as soon as indexing creates one — previously that "missing"
    verdict was permanent for the life of the process.

    Args:
        repo_id: Repository id, or None for the shared/default index.

    Returns:
        The BM25 payload dict (``{"bm25": ..., "chunks": [...]}``), or None
        when no usable index exists.
    """
    path = _bm25_index_path(repo_id)
    stamp = _bm25_stamp(path)

    cached = _bm25_data.get(repo_id)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    if stamp is None:
        # File is absent. Record the miss against the absent-stamp so the
        # next call re-checks once the file appears.
        _bm25_data[repo_id] = (None, None)
        logger.info("BM25 index not found at %s - keyword search disabled.", path)
        return None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        # Raced with a delete between the stat() and the open().
        _bm25_data[repo_id] = (None, None)
        logger.info("BM25 index not found at %s - keyword search disabled.", path)
        return None
    except Exception as e:
        _bm25_data[repo_id] = (stamp, None)
        logger.warning("Failed to load BM25 index at %s: %s", path, e)
        return None

    if cached is not None:
        logger.info("BM25 index at %s changed on disk - reloaded.", path)

    _bm25_data[repo_id] = (stamp, data)
    return data

from query_normalizer import normalize_query

def preprocess_query(query: str) -> str:
    """Normalize user search queries using the query normalization layer."""
    return normalize_query(query)


# ---------------------------------------------------------------------------
# Repository metadata (index cache) — per-repository path, stamped cache
# ---------------------------------------------------------------------------
# Where a pre-multi-repo install kept its index cache. Still the right answer
# when no repository is registered.
LEGACY_METADATA_PATH = ".index_cache.json"

# Parsed `_metadata` blocks keyed by path: {path: (stamp, metadata)} where
# `stamp` is the (mtime_ns, size) of the file it was parsed from, or None when
# the file was missing/unreadable.
_metadata_cache: dict[str, tuple[tuple[int, int] | None, dict]] = {}

# (repo_id, indexed_version, configured_version) triples already warned about,
# so a mismatch produces one line rather than one per query.
_warned_embedding_versions: set[tuple[str | None, str, str]] = set()


def metadata_path_for(repo_id: str | None) -> str:
    """
    Return the index-cache path for *repo_id*.

    Mirrors what ``main.py`` does in ``/statistics`` and ``/index/summary``,
    and what ``indexer.py`` actually writes: per-repository indexes live in
    ``./output/index_cache_<repo_id>.json``, not in the working directory.

    Args:
        repo_id: Repository id, or None when no repository is registered.

    Returns:
        Path to the index cache JSON for that repository.
    """
    if repo_id is None:
        return LEGACY_METADATA_PATH
    return repo_registry.cache_path(repo_id)


def reset_metadata_cache() -> None:
    """Drop the parsed-metadata cache and the warned-version set.

    ``get_repository_metadata()`` re-reads on its own when the file changes;
    this is for tests and for callers that want a clean slate.
    """
    _metadata_cache.clear()
    _warned_embedding_versions.clear()


def get_repository_metadata(metadata_path: str = LEGACY_METADATA_PATH) -> dict:
    """
    Retrieve project-level repository metadata from the indexing cache.

    The parsed ``_metadata`` block is cached against the file's
    ``(mtime_ns, size)`` stamp — the same technique ``_get_bm25()`` uses for
    the BM25 pickle — so a steady-state call pays one stat() instead of an
    ``os.path.exists()`` plus a full ``json.load()``. That matters because
    ``check_embedding_version()`` runs on every single query, and the
    ``_metadata`` block grows with the repository (``skipped_files`` holds one
    dict per skipped file).

    A missing or corrupted file is cached as an empty dict against its stamp,
    so the answer flips to real metadata as soon as indexing writes one.

    Args:
        metadata_path: Path to the index cache JSON. Use
            :func:`metadata_path_for` to derive it from a repository id.

    Returns:
        Dict with metadata (repository_name, indexed_file_count, etc.) or
        empty dict. A shallow copy, so callers cannot mutate the cache.
    """
    stamp = _file_stamp(metadata_path)

    cached = _metadata_cache.get(metadata_path)
    if cached is not None and cached[0] == stamp:
        return dict(cached[1])

    if stamp is None:
        _metadata_cache[metadata_path] = (None, {})
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except FileNotFoundError:
        # Raced with a delete between the stat() and the open().
        _metadata_cache[metadata_path] = (None, {})
        return {}
    except Exception:
        logger.warning("Corrupted repository metadata encountered.")
        _metadata_cache[metadata_path] = (stamp, {})
        return {}

    metadata = {}
    if isinstance(cache_data, dict):
        block = cache_data.get("_metadata")
        if isinstance(block, dict):
            metadata = block

    _metadata_cache[metadata_path] = (stamp, metadata)
    return dict(metadata)


def _tokenize(text: str) -> list[str]:
    """Split code text into lowercased word tokens for BM25."""
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]


def _matches_metadata_filter(payload: dict, metadata_filter: dict | None) -> bool:
    """Check if a document payload satisfies all provided metadata key-value conditions."""
    if not metadata_filter:
        return True
    for key, expected_value in metadata_filter.items():
        if payload.get(key) != expected_value:
            return False
    return True


def _build_qdrant_filter(metadata_filter: dict | None, repository_names: list[str] | None = None) -> qdrant_models.Filter | None:
    """Convert key-value dictionary metadata filters to a Qdrant Filter object."""
    if not metadata_filter and not repository_names:
        return None

    conditions = []
    if metadata_filter:
        for key, value in metadata_filter.items():
            conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=value),
                )
            )
    
    if repository_names:
        if len(repository_names) == 1:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="repository",
                    match=qdrant_models.MatchValue(value=repository_names[0]),
                )
            )
        else:
            # If multiple repos, match any repo in repository_names
            conditions.append(
                qdrant_models.FieldCondition(
                    key="repository",
                    match=qdrant_models.MatchAny(any=repository_names),
                )
            )
    return qdrant_models.Filter(must=conditions) if conditions else None


def _keyword_search(
    query: str,
    top_k: int = HYBRID_TOP_K,
    metadata_filter: dict | None = None,
    repo_id: str | None = None,
    repository_names: list[str] | None = None,
) -> list[dict]:
    """BM25 keyword search filtered by metadata and repositories. Returns chunks with 'bm25_score' and unique '_idx'."""
    bm25_data = _get_bm25(repo_id)
    if bm25_data is None:
        return []

    tokenize_query = _tokenize(query)
    bm25 = bm25_data["bm25"]
    scores = bm25.get_scores(tokenize_query)
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = bm25_data["chunks"][idx].copy()
            # Apply metadata filter
            if not _matches_metadata_filter(chunk, metadata_filter):
                continue
            # Apply repository filter if specified
            if repository_names and chunk.get("repository") and chunk.get("repository") not in repository_names:
                continue

            chunk["bm25_score"] = float(scores[idx])
            chunk["_idx"] = idx
            results.append(chunk)
        if len(results) >= top_k:
            break
    return results


# Question words and other sentence scaffolding that must never be treated as
# code. The tightened CamelCase pattern below already rejects these in their
# usual "How ..." form; this list exists for the all-caps case — someone
# typing "WHERE IS THE PARSER" would otherwise hand WHERE, IS and THE to
# _exact_symbol_search() as acronyms.
_SYMBOL_STOPWORDS = frozenset({
    "how", "what", "where", "why", "when", "which", "who", "whose", "whom",
    "is", "are", "was", "were", "be", "am",
    "does", "do", "did", "done",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "the", "this", "that", "these", "those",
    "and", "or", "not", "but", "if", "then", "else",
    "a", "an", "in", "on", "at", "to", "for", "of", "from", "with", "by",
    "it", "its", "i", "we", "you", "they", "he", "she",
    "show", "tell", "explain", "find", "give", "list", "get", "make", "use",
})

# Words followed by an opening parenthesis — "retrieve()", "get_scores(".
_CALL_SYNTAX_RE = re.compile(r"\b(\w+)\s*\(")

# CamelCase and acronyms. Two alternatives, both requiring more than a single
# leading capital:
#   [A-Z][a-z0-9]*[A-Z]\w*  — an interior capital: DataProcessor, RequestContext
#   [A-Z]{2,}[0-9]*         — an acronym, optionally numbered: BM25, RRF, API
# A single capitalised word ("Where", "Qdrant") is indistinguishable from the
# first word of an English sentence, so it is deliberately not a candidate.
_CAMEL_CASE_RE = re.compile(r"\b([A-Z][a-z0-9]*[A-Z]\w*|[A-Z]{2,}[0-9]*)\b")

# snake_case identifiers — make_repo_id, cache_path, bm25_path.
# The leading segment allows digits so numbered names are caught; the old
# pattern was ``[a-z]+_[a-z_0-9]+``, which required the part before the first
# underscore to be letters only and therefore missed "bm25_path" entirely.
_SNAKE_CASE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")


def _extract_symbols(query: str) -> list[str]:
    """
    Extract possible code symbol names from a natural language query.

    Uses three heuristics:
      1. Words followed by '(' — likely function calls.
      2. CamelCase words and acronyms — likely class names.
      3. snake_case words — likely function/variable names.

    Heuristic 2 used to be ``\\b([A-Z][a-zA-Z0-9]+)\\b``, which asks for a
    leading capital and nothing else. Since users type questions that start
    with a capital letter, it fired on the first word of nearly every query —
    "How", "What", "Where" all came back as class names. Those then went to
    :func:`_exact_symbol_search`, whose fallback branch counts substrings
    across the whole corpus, so the symbol list ended up ranked by how much
    English prose a chunk contained. It now requires an interior capital or a
    genuine acronym.

    Args:
        query: User's natural language query.

    Returns:
        Unique symbol name candidates, in the order they were found. The
        ordering is deliberate — this used to return ``list(set(...))``,
        whose order varies between runs and makes the caller hard to test.
    """
    symbols: dict[str, None] = {}

    for pattern in (_CALL_SYNTAX_RE, _CAMEL_CASE_RE, _SNAKE_CASE_RE):
        for match in pattern.finditer(query):
            candidate = match.group(1)
            if candidate.lower() in _SYMBOL_STOPWORDS:
                continue
            symbols.setdefault(candidate, None)

    return list(symbols)


def _exact_symbol_search(
    symbols: list[str],
    top_k: int = HYBRID_TOP_K,
    metadata_filter: dict | None = None,
    repo_id: str | None = None,
    repository_names: list[str] | None = None,
) -> list[dict]:
    """
    Find chunks with exact symbol name matches, filtered by metadata and repositories, ranked by match count.
    """

    # Checked before touching the index: with the symbol list now correctly
    # empty for plain-English questions, this is the common case and there is
    # no reason to load a corpus we are about to scan zero symbols against.
    if not symbols:
        return []

    bm25_data = _get_bm25(repo_id)
    if bm25_data is None:
        return []

    # Compiled once for the whole corpus scan rather than per chunk.
    # \b anchors each symbol to a word boundary, so searching for "get" stops
    # scoring chunks that merely contain "budget", "target" or "forget".
    # Python treats "_" as a word character, so snake_case names are matched
    # whole: "\bcache_path\b" does not fire on "cache_path_for".
    symbol_patterns = [
        re.compile(r"\b" + re.escape(sym.lower()) + r"\b") for sym in symbols
    ]

    matches = []
    for idx, chunk in enumerate(bm25_data["chunks"]):
        if metadata_filter and not _matches_metadata_filter(chunk, metadata_filter):
            continue
        if repository_names and chunk.get("repository") and chunk.get("repository") not in repository_names:
            continue

        chunk_name = chunk.get("symbol_name")
        is_symbol = chunk.get("is_symbol", False)

        if is_symbol and chunk_name:
            count = sum(
                1 for sym in symbols if chunk_name.lower() == sym.lower()
            )
        else:
            text_lower = chunk["text"].lower()
            count = sum(len(pattern.findall(text_lower)) for pattern in symbol_patterns)

        if count > 0:
            result = chunk.copy()
            result["exact_match_count"] = count
            result["_idx"] = f"s_{idx}"
            matches.append(result)

    matches.sort(key=lambda x: -x["exact_match_count"])
    return matches[:top_k]


# Fields each retriever contributes on its own and the others leave unset.
# When two retrievers turn out to have found the same chunk, the merged copy
# should carry all of them — otherwise the confidence table in retrieve()
# misses the dense score on a chunk that BM25 happened to rank first.
_RETRIEVER_SCORE_FIELDS = ("score", "bm25_score", "exact_match_count")


def _fusion_key(doc: dict) -> tuple:
    """
    Return a content identity for *doc*, used to group it across ranked lists.

    The three retrievers number their results independently — Qdrant results
    are ``v_<position in the response>``, BM25 results are the integer offset
    into the pickled corpus, symbol results are ``s_<same offset>``. Those
    namespaces never overlap, so grouping on ``_idx`` puts one physical chunk
    into as many buckets as the retrievers that found it. Grouping on where
    the chunk actually lives in the repository puts it into one.

    Args:
        doc: A chunk dict from any of the three retrievers.

    Returns:
        A hashable identity. Falls back to the retriever-local ``_idx`` for
        chunks with no usable file path, so results that carry no location
        (older payloads, hand-built test fixtures) still fuse by identity
        rather than collapsing into each other.
    """
    file = doc.get("file")
    if file and file != "unknown":
        return (
            "chunk",
            doc.get("repository") or "",
            file,
            doc.get("start_line"),
            doc.get("symbol_name") or "",
        )
    return ("idx", str(doc.get("_idx", id(doc))))


def _merge_retriever_fields(target: dict, source: dict) -> None:
    """
    Copy the score fields *source* has and *target* lacks onto *target*.

    Called when two retrievers return the same chunk. Existing values always
    win, so the first list to report a chunk keeps its own numbers and only
    genuinely missing fields are filled in.

    Args:
        target: The chunk copy kept in the fused output. Mutated in place.
        source: A duplicate of the same chunk from another ranked list.
    """
    for field in _RETRIEVER_SCORE_FIELDS:
        if target.get(field) is None and source.get(field) is not None:
            target[field] = source[field]


def _rrf_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
    final_top_k: int = HYBRID_TOP_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — combine multiple ranked result lists by position.

    RRF score = Σ 1 / (k + rank + 1) for each document across all lists.
    Documents appearing in multiple lists get boosted.

    Documents are grouped by :func:`_fusion_key` rather than by the ``_idx``
    each retriever assigns, so a chunk that dense search, BM25 and symbol
    matching all found is scored once with all three contributions instead of
    three times with one contribution each. That is what makes the "appears in
    multiple lists" boost real, and it stops the same code being handed to the
    LLM two or three times over inside a single ``top_k`` budget.

    A document is only credited once per list. Duplicates within one list —
    two chunks of the same function, say — would otherwise let a single
    retriever compound its own vote.

    Args:
        result_lists: List of ranked result lists (each a list of chunk dicts).
        k: RRF constant (default 60) — dampens the influence of low ranks.
        final_top_k: Number of top-fused results to return.

    Returns:
        List of chunk dicts augmented with 'rrf_score', sorted by score
        descending. Ties are broken by first appearance, so the ordering is
        stable across runs.
    """
    scores: dict[tuple, float] = {}
    doc_map: dict[tuple, dict] = {}
    order: dict[tuple, int] = {}

    for results in result_lists:
        seen_in_list: set[tuple] = set()
        for rank, doc in enumerate(results):
            key = _fusion_key(doc)
            if key in seen_in_list:
                # Same chunk twice in one ranked list — one vote, not two.
                continue
            seen_in_list.add(key)

            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc.copy()
                order[key] = len(order)
            else:
                _merge_retriever_fields(doc_map[key], doc)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
    final = []
    for key, score in ranked[:final_top_k]:
        doc = doc_map[key]
        doc["rrf_score"] = score
        final.append(doc)
    return final


def check_embedding_version(repo_id: str | None = None) -> None:
    """
    Verify that the embedding version in the index matches the configured version.

    Logs a warning if a mismatch is detected, advising re-indexing.

    This used to read ``.index_cache.json`` unconditionally, which is not
    where a registered repository's metadata lives — ``indexer.py`` writes to
    ``./output/index_cache_<repo_id>.json``. So the one diagnostic built to
    catch "you changed the embedding model and did not re-index" never fired
    for any repository, and if a legacy ``.index_cache.json`` happened to be
    lying around it reported a clean bill of health for a repository it had
    never looked at.

    A mismatch is warned about once per (repository, version pair) rather
    than on every query, since this runs on the hot path.

    Args:
        repo_id: Repository whose index metadata to check. None reads the
            legacy working-directory cache.
    """
    metadata = get_repository_metadata(metadata_path_for(repo_id))
    if not metadata:
        return

    repo_version = metadata.get("embedding_version")
    if not repo_version or repo_version == EMBEDDING_VERSION:
        return

    seen_key = (repo_id, repo_version, EMBEDDING_VERSION)
    if seen_key in _warned_embedding_versions:
        return
    _warned_embedding_versions.add(seen_key)

    logger.warning(
        f"Embedding version mismatch detected (repository version: {repo_version}, "
        f"configured version: {EMBEDDING_VERSION}). Re-indexing is recommended."
    )


from pipeline_hooks import hook_registry


def _resolve_request_context(
    query: str | RequestContext,
    repo_id: str | None,
    context: RequestContext | None,
) -> tuple[str, str | None, RequestContext | None]:
    """Unwrap a RequestContext passed positionally or via ``context=``.

    Mirrors how ``llm.generate_response()`` handles the same two calling
    styles. Explicit arguments always win over values carried on the
    context, so an existing caller that passes both keeps its behaviour.

    Returns:
        (query_text, repo_id, request_context)
    """
    if isinstance(query, RequestContext):
        context = query
        query = ""

    if context is not None:
        if not query:
            query = context.user_query or ""
        if repo_id is None:
            repo_id = context.repo_id

    return query, repo_id, context


def _fuse_ranked_lists(
    vector_chunks: list[dict],
    keyword_chunks: list[dict],
    symbol_chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Combine the three ranked lists into the final result set.

    Rules:
        - Two or three non-empty lists → Reciprocal Rank Fusion.
        - Exactly one non-empty list → return that list, whichever it is.
        - No non-empty lists → empty result.

    The middle case is the one that used to be wrong. The old code fell back
    to ``vector_chunks[:top_k]`` by name, which is only correct when the one
    surviving list *is* the vector one. When dense search came back empty and
    BM25 had matches, that fallback returned ``[]`` and the whole query was
    answered from an empty context — see issue #267.

    Dense search comes back empty more often than it looks: a query phrased
    below ``QDRANT_SIMILARITY_THRESHOLD`` returns no points at all, the
    collection for the active repository may not exist yet while its BM25
    pickle does, and a ``metadata_filter`` can exclude everything Qdrant
    found. In each of those cases the sparse retrievers still hold the answer.

    Args:
        vector_chunks: Dense search results, best-first.
        keyword_chunks: BM25 results, best-first.
        symbol_chunks: Exact symbol matches, best-first.
        top_k: Maximum number of results to return.

    Returns:
        The fused (or single-list) results, truncated to *top_k*.
    """
    ranked_lists = [
        results
        for results in (vector_chunks, keyword_chunks, symbol_chunks)
        if results
    ]

    if not ranked_lists:
        return []

    if len(ranked_lists) == 1:
        if not vector_chunks:
            # Worth a line in the log: retrieval succeeded on the sparse side
            # only, which usually means the dense threshold is too aggressive
            # for this phrasing or the collection is missing.
            logger.debug(
                "Dense search returned nothing; answering from %d sparse result(s).",
                len(ranked_lists[0]),
            )
        return ranked_lists[0][:top_k]

    return _rrf_fusion(ranked_lists, final_top_k=top_k)


def retrieve(
    query: str | RequestContext = "",
    top_k: int = RETRIEVAL_TOP_K,
    include_sources: bool = False,
    metadata_filter: dict | None = None,
    repo_id: str | None = None,
    repositories: list[str] | str | None = None,
    context: RequestContext | None = None,
):
    """
    Hybrid retrieval with pipeline hooks execution.

    Args:
        query: The search query, or a ``RequestContext`` carrying one.
        top_k: Number of fused results to return.
        include_sources: Return the sources and confidence tables alongside
            the formatted context.
        metadata_filter: Optional payload key/value equality filters.
        repo_id: Repository whose index to search. Falls back to the id on
            *context* when not given explicitly.
        repositories: Repository name(s) to restrict shared-collection results to.
        context: Optional ``RequestContext`` (alternative to passing it as *query*).

    Returns:
        ``(formatted_context, sources, confidences)`` when *include_sources*
        is true, otherwise the formatted context string. The three-value
        shape is returned on every path, including the empty-query
        early return — ``main.py`` unpacks three values unconditionally.
    """
    query, repo_id, context = _resolve_request_context(query, repo_id, context)

    hook_registry.execute_pre_hooks("retrieval", {"query": query, "top_k": top_k, "repo_id": repo_id})
    if repo_id is not None:
        target_collection = repo_registry.collection_name(repo_id)
    else:
        target_collection = QDRANT_COLLECTION_NAME
    query = preprocess_query(query)
    if not query:
        return ("", [], {}) if include_sources else ""

    check_embedding_version(repo_id)

    # Normalize repositories parameter into a list if provided
    repo_list = None
    if isinstance(repositories, str):
        repo_list = [repositories]
    elif isinstance(repositories, list):
        repo_list = repositories

    # ── Dense vector search (Qdrant) ────────────────────────────────────
    vector = embedder.encode(query).tolist()
    query_limit = HYBRID_TOP_K if _get_bm25(repo_id) is not None else top_k
    # Repo-isolated collections already scope results to one repository, and
    # legacy payloads have no ``repository`` tag — so only apply the
    # repository-name filter in shared-collection mode (repo_id is None).
    qdrant_filter = _build_qdrant_filter(metadata_filter, repo_list if repo_id is None else None)

    qdrant_result = client.query_points(
        collection_name=target_collection,
        query=vector,
        query_filter=qdrant_filter,
        limit=query_limit,
        score_threshold=QDRANT_SIMILARITY_THRESHOLD,
    )

    # qdrant_result may be an object with a .points attribute (Qdrant client) or
    # a simple iterable. Normalize to an iterable of points for testability.
    points_iterable = getattr(qdrant_result, "points", qdrant_result)

    # Expose the last query vector into builtins so older tests that reference
    # the name `vector` directly (unqualified) can still assert against it.
    try:
        import builtins as _builtins
        _builtins.vector = vector
    except Exception:
        pass

    vector_chunks = []
    for idx, point in enumerate(points_iterable):
        payload = point.payload or {}
        repo_name = payload.get("repository", "")
        vector_chunks.append({
            "_idx": f"v_{idx}",
            "text": payload.get("text", ""),
            "file": payload.get("file", "unknown"),
            "repository": repo_name,
            "start_line": payload.get("start_line", "?"),
            "end_line": payload.get("end_line"),
            "symbol_name": payload.get("symbol_name"),
            "symbol_type": payload.get("symbol_type"),
            "parent_class": payload.get("parent_class"),
            "docstring": payload.get("docstring"),
            "is_symbol": payload.get("is_symbol", False),
            "score": getattr(point, "score", None)
        })

    # ── Sparse keyword search (BM25) ────────────────────────────────────
    keyword_chunks = _keyword_search(query, HYBRID_TOP_K, metadata_filter=metadata_filter, repo_id=repo_id, repository_names=repo_list)

    # ── Exact symbol matching ──────────────────────────────────────────────
    symbols = _extract_symbols(query)
    symbol_chunks = _exact_symbol_search(symbols, HYBRID_TOP_K, metadata_filter=metadata_filter, repo_id=repo_id, repository_names=repo_list)

    # ── Fuse results ──────────────────────────────────────────────────────
    fused = _fuse_ranked_lists(vector_chunks, keyword_chunks, symbol_chunks, top_k)

    # ── Format context for LLM ──────────────────────────────────────────
    structured_context = []
    sources = []
    confidences = {}
    for index, result in enumerate(fused):

        confidence = result.get("score")

        file = result.get("file", "unknown")
        repo = result.get("repository", "")
        start_line = result.get("start_line", "?")
        code = result.get("text", "")

        # Distinct source identification with repository and file path
        source_label = f"{repo}:{file}" if repo else file
        if source_label and source_label != "unknown":
            sources.append(source_label)
            # Results are ordered best-first and `sources` is de-duplicated
            # keeping the first occurrence, so the confidence table has to
            # keep the first (highest ranked) score for a label too — not
            # let a later, weaker chunk from the same file overwrite it.
            if source_label not in confidences:
                confidences[source_label] = (
                    round(confidence * 100) if confidence is not None else None
                )

        symbol_name = result.get("symbol_name")
        symbol_type = result.get("symbol_type")
        parent_class = result.get("parent_class")
        docstring = result.get("docstring")
        end_line = result.get("end_line")

        if symbol_name:
            if parent_class:
                display_name = f"{parent_class}.{symbol_name}"
            else:
                display_name = symbol_name
            entity_label = symbol_type.capitalize() if symbol_type else "Symbol"
        else:
            entity_label = "Function"
            display_name = "unknown"
            for line in code.split("\n"):
                if line.strip().startswith("def "):
                    display_name = (
                        line.strip().split("(")[0].replace("def ", "")
                    )
                    break

        location = f"Line {start_line}"
        if end_line and end_line != start_line:
            location = f"Lines {start_line}-{end_line}"

        doc_block = ""
        if docstring:
            doc_block = f"Docstring: {docstring}\n"

        repo_tag = f"Repository: {repo}\n" if repo else ""
        structured_context.append(
            f"""Result {index + 1}:
{repo_tag}File: {file}
{entity_label}: {display_name}
Location: {location}
{doc_block}Code:
{code}
"""
        )

    formatted_context = "\n\n".join(structured_context)
    unique_sources = list(dict.fromkeys(sources))
    if include_sources:
        # Keep the confidence table aligned with the sources actually
        # returned, so callers can look up every source without a KeyError.
        result = (
            formatted_context,
            unique_sources,
            {label: confidences.get(label) for label in unique_sources},
        )
    else:
        result = formatted_context
    hook_registry.execute_post_hooks("retrieval", {"query": query, "top_k": top_k, "repo_id": repo_id}, result)
    return result
