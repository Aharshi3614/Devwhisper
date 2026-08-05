"""
handlers.py — Command routing and request processing pipeline for DevWhisper.

This module provides command routing and an explicit, modular request processing
pipeline to handle query execution stages:
    1. Validation
    2. Cache lookup
    3. Retrieval (hybrid search)
    4. Command routing / LLM generation
    5. Cache insertion & session memory update
"""

from logger import logger
from cache import get as cache_get, put as cache_put
from retriever import retrieve
from llm import generate_response


def route_command(query: str, session_id: str) -> str | None:
    """
    Route special-case queries to pre-defined handlers.

    Checks the normalized query against known command patterns. If matched,
    returns a direct response string bypassing the retrieval + LLM pipeline.
    If no match, returns None so the caller can fall back to standard processing.
    """
    normalized = query.strip().lower()

    # Help command
    if normalized in ("help", "what can you do", "what can you do?"):
        return (
            "I can help you understand your codebase. Ask me things like:\n"
            "- 'What does the preprocess function do?'\n"
            "- 'Where is the model saved after training?'\n"
            "- 'How do I debug a KeyError in the pipeline?'"
        )

    # Status command
    if normalized in ("status", "are you working", "are you working?"):
        return "DevWhisper is online and ready to help with your codebase."

    # No match — fall back to standard retrieval + LLM pipeline
    return None


def process_query_pipeline(query: str, session_id: str, memory_getter, memory_updater, repositories=None) -> tuple[str, list[str]]:
    """
    Execute the explicit request processing pipeline stages:
        Stage 1: Cache Lookup
        Stage 2: Retrieval (Hybrid Search)
        Stage 3: Command Routing or LLM Generation
        Stage 4: Post-processing & Attribution
        Stage 5: Cache Insertion & Memory Update

    Args:
        query: User query string.
        session_id: Active session identifier.
        memory_getter: Callable to fetch conversation history.
        memory_updater: Callable to update session history.
        repositories: Optional repository selection list.

    Returns:
        Tuple of (final_answer_string, list_of_sources).
    """
    # Stage 1: Cache Lookup
    cached = cache_get(query)
    if cached is not None:
        memory_updater(session_id, query, cached)
        logger.info("Cache hit for query: %s", query)
        return cached, []

    # Stage 2: Retrieval
    context, sources = retrieve(query, include_sources=True, repositories=repositories)
    history = memory_getter(session_id)

    # Stage 3: Command Routing / LLM Generation
    answer = route_command(query, session_id)
    if not answer:
        answer = generate_response(query, context, history)

    # Stage 4: Post-processing & Source Attribution
    if answer and answer.strip() and sources:
        answer += "\n\n**Sources used:** " + ", ".join(f"`{s}`" for s in sources)

    # Stage 5: Cache Insertion & Memory Update
    if answer and answer.strip():
        cache_put(query, answer)
    
    memory_updater(session_id, query, answer)

    return answer, sources
