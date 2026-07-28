import json
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    QDRANT_API_KEY_ENV,
    QDRANT_COLLECTION_NAME,
    QDRANT_SIMILARITY_THRESHOLD,
    QDRANT_URL_ENV,
    RETRIEVAL_TOP_K,
)
from logger import logger

client = QdrantClient(
    url=os.getenv(QDRANT_URL_ENV),
    api_key=os.getenv(QDRANT_API_KEY_ENV),
)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def get_repository_metadata(metadata_path: str = ".index_cache.json") -> dict:
    """Retrieve project-level repository metadata."""
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception:
        # Gracefully handle corrupted metadata
        logger.warning("Corrupted repository metadata encountered.")
        return {}

    if isinstance(cache_data, dict):
        metadata = cache_data.get("_metadata")
        if isinstance(metadata, dict):
            return metadata
    return {}


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


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    include_sources: bool = False,
):
    """Retrieve the most relevant code snippets for a natural-language query.

    Encodes ``query`` into an embedding, performs a Qdrant vector search, and
    formats the top matches into a human-readable context string.
    """
    check_embedding_version()
    vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=vector,
        limit=top_k,
        score_threshold=QDRANT_SIMILARITY_THRESHOLD,
    ).points

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
