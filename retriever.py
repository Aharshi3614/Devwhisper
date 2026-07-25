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
