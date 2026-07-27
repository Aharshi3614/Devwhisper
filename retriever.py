import difflib
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY_ENV,
    QDRANT_COLLECTION_NAME,
    QDRANT_SIMILARITY_THRESHOLD,
    QDRANT_URL_ENV,
    RETRIEVAL_TOP_K,
)

client = QdrantClient(
    url=os.getenv(QDRANT_URL_ENV),
    api_key=os.getenv(QDRANT_API_KEY_ENV),
)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def _parse_line_range(start_line, code: str):
    """Parse start line and calculate range based on line count."""
    if start_line is None or start_line == "?":
        return None
    try:
        start_val = int(start_line)
    except (ValueError, TypeError):
        return None
    num_lines = len(code.splitlines())
    return (start_val, start_val + max(0, num_lines - 1))


def _is_duplicate_or_similar(a_payload: dict, a_code: str, b_payload: dict, b_code: str) -> bool:
    """Determine if chunk b is a duplicate, overlap, or highly similar to chunk a."""
    # 1. Normalize whitespace for standard comparison
    norm_a = " ".join(a_code.split())
    norm_b = " ".join(b_code.split())
    if norm_a == norm_b:
        return True

    file_a = a_payload.get("file")
    file_b = b_payload.get("file")

    # 2. Check same file characteristics (overlap / substring)
    if file_a and file_b and file_a != "unknown" and file_a == file_b:
        if norm_a in norm_b or norm_b in norm_a:
            return True

        range_a = _parse_line_range(a_payload.get("start_line"), a_code)
        range_b = _parse_line_range(b_payload.get("start_line"), b_code)
        if range_a and range_b:
            s1, e1 = range_a
            s2, e2 = range_b
            if max(s1, s2) <= min(e1, e2):
                return True

    # 3. Fuzzy similarity threshold (0.85) to catch nearly identical modifications
    if difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.85:
        return True

    return False


def deduplicate_chunks(results):
    """Filter out duplicate, highly similar, or overlapping retrieved chunks.

    Preserves only the highest-ranked chunk among duplicates or overlapping candidates
    and maintains the original ranking order.
    """
    deduplicated = []
    for result in results:
        payload_b = result.payload or {}
        code_b = payload_b.get("text", "")

        is_dup = False
        for seen in deduplicated:
            payload_a = seen.payload or {}
            code_a = payload_a.get("text", "")
            if _is_duplicate_or_similar(payload_a, code_a, payload_b, code_b):
                is_dup = True
                break

        if not is_dup:
            deduplicated.append(result)
    return deduplicated


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    include_sources: bool = False,
):
    """Retrieve the most relevant code snippets for a natural-language query.

    Encodes ``query`` into an embedding, performs a Qdrant vector search, and
    formats the top matches into a human-readable context string.
    """
    vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=vector,
        limit=top_k,
        score_threshold=QDRANT_SIMILARITY_THRESHOLD,
    ).points

    # Deduplicate results before assembling structured context
    results = deduplicate_chunks(results)

    structured_context = []
    sources = []
    for index, result in enumerate(results):
        payload = result.payload or {}
        file = payload.get("file", "unknown")
        start_line = payload.get("start_line", "?")
        code = payload.get("text", "")

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
