import json
import os
import pickle
import re

from sentence_transformers import SentenceTransformer
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

client = vector_store.client
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

# Load BM25 index for hybrid retrieval
_bm25_data = None
try:
    with open(BM25_INDEX_PATH, "rb") as f:
        _bm25_data = pickle.load(f)
    logger.info(f"BM25 index loaded from {BM25_INDEX_PATH}")
except FileNotFoundError:
    logger.info("BM25 index not found - keyword search disabled")
except Exception as e:
    logger.warning(f"Failed to load BM25 index: {e}")


def preprocess_query(query: str) -> str:
    """Normalize user search queries by stripping whitespace and redundant punctuation."""
    if not query:
        return ""

    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r'^[^\w\s()_.\-]+|[^\w\s()_.\-]+$', "", query).strip()
    return query


def get_repository_metadata(metadata_path: str = ".index_cache.json") -> dict:
    """Retrieve project-level repository metadata."""
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
    """Split code text into lowercased word tokens for BM25"""
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]


def _matches_metadata_filter(payload: dict, metadata_filter: dict | None) -> bool:
    """Check if a document payload satisfies all provided metadata key-value conditions."""
    if not metadata_filter:
        return True
    for key, expected_value in metadata_filter.items():
        if payload.get(key) != expected_value:
            return False
    return True


def _build_qdrant_filter(metadata_filter: dict | None):
    """Convert key-value dictionary metadata filters to a Qdrant Filter object."""
    if not metadata_filter:
        return None

    conditions = []
    for key, value in metadata_filter.items():
        conditions.append(
            vector_store.qdrant_models.FieldCondition(
                key=key,
                match=vector_store.qdrant_models.MatchValue(value=value),
            )
        )
    return vector_store.qdrant_models.Filter(must=conditions) if conditions else None


def _keyword_search(
    query: str,
    top_k: int = HYBRID_TOP_K,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """BM25 keyword search filtered by metadata. Returns chunks with 'bm25_score' and unique '_idx'."""
    if _bm25_data is None:
        return []
    tokenize_query = _tokenize(query)
    bm25 = _bm25_data["bm25"]
    scores = bm25.get_scores(tokenize_query)
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = _bm25_data["chunks"][idx].copy()
            if _matches_metadata_filter(chunk, metadata_filter):
                chunk["bm25_score"] = float(scores[idx])
                chunk["_idx"] = idx
                results.append(chunk)
                if len(results) >= top_k:
                    break
    return results


def _extract_symbols(query: str) -> list[str]:
    """Extract possible code symbol names from a natural language query."""
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
) -> list[dict]:
    """Find chunks containing exact symbol name matches, filtered by metadata, ranked by match count."""
    if not symbols or _bm25_data is None:
        return []
    matches = []
    for idx, chunk in enumerate(_bm25_data["chunks"]):
        if not _matches_metadata_filter(chunk, metadata_filter):
            continue
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
    """Reciprocal Rank Fusion - fuse multiple ranked result lists by position."""
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


def check_embedding_version():
    """Verify that the embedding version matches the configured version."""
    metadata = get_repository_metadata()
    if metadata:
        repo_version = metadata.get("embedding_version")
        if repo_version and repo_version != EMBEDDING_VERSION:
            logger.warning(
                f"Embedding version mismatch detected (repository version: {repo_version}, "
                f"configured version: {EMBEDDING_VERSION}). Re-indexing is recommended."
            )

def _vector_search(
    query: str,
    top_k: int,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Perform vector search and return formatted vector chunks."""
    vector = embedder.encode(query).tolist()
    query_limit = HYBRID_TOP_K if _bm25_data is not None else top_k
    qdrant_filter = _build_qdrant_filter(metadata_filter)

    qdrant_result = vector_store.query_points(
        vector=vector,
        limit=query_limit,
        query_filter=qdrant_filter,
        collection_name=QDRANT_COLLECTION_NAME,
        score_threshold=QDRANT_SIMILARITY_THRESHOLD,
    )

    vector_chunks = []
    for idx, point in enumerate(qdrant_result):
        payload = point.payload or {}
        vector_chunks.append(
            {
                "_idx": f"v_{idx}",
                "text": payload.get("text", ""),
                "file": payload.get("file", "unknown"),
                "start_line": payload.get("start_line", "?"),
                **payload,
            }
        )

    return vector_chunks

    
def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    include_sources: bool = False,
    metadata_filter: dict | None = None,
):
    """Hybrid retrieval: vector + BM25 + exact symbol matching fused via RRF with preprocessing & metadata filtering."""
    check_embedding_version()

    query = preprocess_query(query)
    if not query:
        return ("", []) if include_sources else ""

    vector_chunks = _vector_search(query, top_k, metadata_filter)

    keyword_chunks = _keyword_search(query, HYBRID_TOP_K, metadata_filter=metadata_filter)

    symbols = _extract_symbols(query)
    symbol_chunks = _exact_symbol_search(symbols, HYBRID_TOP_K, metadata_filter=metadata_filter)

    all_nonempty_chunks = [r for r in [vector_chunks, keyword_chunks, symbol_chunks] if r]
    if len(all_nonempty_chunks) > 1:
        fused = _rrf_fusion(all_nonempty_chunks, final_top_k=top_k)
    else:
        fused = vector_chunks[:top_k]

    structured_context = []
    sources = []
    for index, result in enumerate(fused):
        file = result.get("file", "unknown")
        start_line = result.get("start_line", "?")
        code = result.get("text", "")

        if file and file != "unknown":
            sources.append(file)

        function_name = "unknown"
        for line in code.split("\n"):
            if line.strip().startswith("def "):
                function_name = (
                    line.strip().split("(")[0].replace("def ", "")
                )
                break

        structured_context.append(
            f"""
Result {index + 1}:
File: {file}
Function: {function_name}
Start Line: {start_line}
Code:
{code}
"""
        )

    formatted_context = "\n\n".join(structured_context)
    if include_sources:
        unique_sources = list(dict.fromkeys(sources))
        return formatted_context, unique_sources
    return formatted_context
