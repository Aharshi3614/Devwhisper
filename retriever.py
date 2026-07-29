"""Qdrant-backed code retrieval helpers."""

from dependencies import RetrievalDependencies, get_retrieval_dependencies
from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
)


def _resolve_dependencies(
    dependencies: RetrievalDependencies | None,
) -> RetrievalDependencies:
    """Use injected dependencies when provided, otherwise resolve defaults."""
    return dependencies or get_retrieval_dependencies()


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    include_sources: bool = False,
    dependencies: RetrievalDependencies | None = None,
):
    """Retrieve the most relevant code snippets for a natural-language query."""
    resolved = _resolve_dependencies(dependencies)
    vector = resolved.embedder.encode(query).tolist()
    results = resolved.client.query_points(
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
