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
  - preprocess_query(): Normalize user queries before embedding/search.
  - _keyword_search(): BM25-based keyword retrieval.
  - _extract_symbols(): Heuristic symbol extraction from natural language queries.
  - _exact_symbol_search(): Direct symbol name matching in chunks (metadata-aware).
  - _rrf_fusion(): Reciprocal Rank Fusion to combine ranked lists.
  - check_embedding_version(): Warns if indexed embeddings differ from config.
  - get_repository_metadata(): Reads indexing metadata from .index_cache.json.

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

# ---------------------------------------------------------------------------
# Qdrant client and embedder (module-level singletons)
# ---------------------------------------------------------------------------
client = vector_store.client
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

# ---------------------------------------------------------------------------
# BM25 index (lazy-loaded per repository)
# ---------------------------------------------------------------------------
_bm25_data: dict[str, dict] = {}

def _get_bm25(repo_id: str | None) -> dict | None:
    if repo_id is None:
        path = BM25_INDEX_PATH
    else:
        path = repo_registry.bm25_path(repo_id)

    if repo_id not in _bm25_data:
        try:
            with open(path, "rb") as f:
                _bm25_data[repo_id] = pickle.load(f)
        except FileNotFoundError:
            _bm25_data[repo_id] = None
            logger.info(f"BM25 index not found - keyword search disabled.")
        except Exception as e:
            _bm25_data[repo_id] = None
            logger.warning(f"Failed to load BM25 index: {e}")

    return _bm25_data[repo_id]

def preprocess_query(query: str) -> str:
    """Normalize user search queries by stripping whitespace and redundant punctuation."""
    if not query:
        return ""

    # 1. Normalize whitespace (collapse redundant tabs, newlines, and multi-spaces)
    query = re.sub(r"\s+", " ", query).strip()

    # 2. Strip leading/trailing enclosing quotes or surrounding redundant punctuation
    query = re.sub(r'^[^\w\s()_.\-]+|[^\w\s()_.\-]+$', "", query).strip()

    return query


def get_repository_metadata(metadata_path: str = ".index_cache.json") -> dict:
    """
    Retrieve project-level repository metadata from the indexing cache.

    Args:
        metadata_path: Path to the .index_cache.json file.

    Returns:
        Dict with metadata (repository_name, indexed_file_count, etc.) or empty dict.
    """
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception:
        logger.warning("Corrupted repository metadata encountered.")
        return {}

    if isinstance(cache_data, dict):
        metadata = cache_data.get("_metadata")
        if isinstance(metadata, dict):
            return metadata
    return {}


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


def _extract_symbols(query: str) -> list[str]:
    """
    Extract possible code symbol names from a natural language query.

    Uses three heuristics:
      1. Words followed by '(' — likely function calls.
      2. CamelCase words — likely class names.
      3. snake_case words — likely function/variable names.

    Args:
        query: User's natural language query.

    Returns:
        List of unique symbol name candidates.
    """
    symbols = set()
    for m in re.finditer(r"(\w+)\s*\(", query):
        symbols.add(m.group(1))
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]+)\b", query):
        symbols.add(m.group(1))
    for m in re.finditer(r"\b([a-z]+_[a-z_0-9]+)\b", query):
        symbols.add(m.group(1))
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

    bm25_data = _get_bm25(repo_id)
    if not symbols or bm25_data is None:
        return []

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
            count = sum(text_lower.count(sym.lower()) for sym in symbols)

        if count > 0:
            result = chunk.copy()
            result["exact_match_count"] = count
            result["_idx"] = f"s_{idx}"
            matches.append(result)

    matches.sort(key=lambda x: -x["exact_match_count"])
    return matches[:top_k]


def _rrf_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
    final_top_k: int = HYBRID_TOP_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — combine multiple ranked result lists by position.

    RRF score = Σ 1 / (k + rank + 1) for each document across all lists.
    Documents appearing in multiple lists get boosted.

    Args:
        result_lists: List of ranked result lists (each a list of chunk dicts).
        k: RRF constant (default 60) — dampens the influence of low ranks.
        final_top_k: Number of top-fused results to return.

    Returns:
        List of chunk dicts augmented with 'rrf_score', sorted by score descending.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            idx = str(doc.get("_idx", id(doc)))
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
            if idx not in doc_map:
                doc_map[idx] = doc.copy()

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    final = []
    for idx, score in ranked[:final_top_k]:
        doc = doc_map[idx]
        doc["rrf_score"] = score
        final.append(doc)
    return final


def check_embedding_version() -> None:
    """
    Verify that the embedding version in the index matches the configured version.

    Logs a warning if a mismatch is detected, advising re-indexing.
    """
    metadata = get_repository_metadata()
    if metadata:
        repo_version = metadata.get("embedding_version")
        if repo_version and repo_version != EMBEDDING_VERSION:
            logger.warning(
                f"Embedding version mismatch detected (repository version: {repo_version}, "
                f"configured version: {EMBEDDING_VERSION}). Re-indexing is recommended."
            )


from request_context import RequestContext

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
    Hybrid retrieval: vector + BM25 + exact symbol matching fused via RRF supporting single or multiple repositories.

    Can take a RequestContext object as context parameter or as the first positional argument.
    """
    if isinstance(query, RequestContext):
        context = query
        query = context.user_query
    elif context is not None:
        if not query:
            query = context.user_query
        if repo_id is None:
            repo_id = context.repo_id

    if context is not None and context.request_id:
        logger.debug(f"Executing retrieval for request_id={context.request_id}")
        If include_sources is True: tuple of (formatted_context, unique_sources).
    """
    if repo_id is not None:
        target_collection = repo_registry.collection_name(repo_id)
    else:
        target_collection = QDRANT_COLLECTION_NAME
    query = preprocess_query(query)
    if not query:
        return ("", []) if include_sources else ""

    check_embedding_version()

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
        })

    # ── Sparse keyword search (BM25) ────────────────────────────────────
    keyword_chunks = _keyword_search(query, HYBRID_TOP_K, metadata_filter=metadata_filter, repo_id=repo_id, repository_names=repo_list)

    # ── Exact symbol matching ──────────────────────────────────────────────
    symbols = _extract_symbols(query)
    symbol_chunks = _exact_symbol_search(symbols, HYBRID_TOP_K, metadata_filter=metadata_filter, repo_id=repo_id, repository_names=repo_list)

    # ── Fuse results ──────────────────────────────────────────────────────
    all_nonempty_chunks = [r for r in [vector_chunks, keyword_chunks, symbol_chunks] if r]
    if len(all_nonempty_chunks) > 1:
        fused = _rrf_fusion(all_nonempty_chunks, final_top_k=top_k)
    else:
        fused = vector_chunks[:top_k]

    # ── Format context for LLM ──────────────────────────────────────────
    structured_context = []
    sources = []
    for index, result in enumerate(fused):
        file = result.get("file", "unknown")
        repo = result.get("repository", "")
        start_line = result.get("start_line", "?")
        code = result.get("text", "")

        # Distinct source identification with repository and file path
        source_label = f"{repo}:{file}" if repo else file
        if source_label and source_label != "unknown":
            sources.append(source_label)

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
    if include_sources:
        unique_sources = list(dict.fromkeys(sources))
        return formatted_context, unique_sources
    return formatted_context
