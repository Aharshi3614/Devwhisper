"""
handlers.py — Command routing and special-case handling for DevWhisper queries.

This module provides a lightweight command router that intercepts specific
user queries before they reach the standard retrieval + LLM pipeline. It is
useful for handling:

    - Built-in commands (e.g., "help", "reset", "status")
    - Administrative operations
    - Shortcut responses that don't require LLM generation

The route_command() function is called by the webhook handler in main.py
before falling back to the full retrieval pipeline.

Usage:
    from handlers import route_command
    answer = route_command(query, session_id)
    if answer:
        # Use the routed response directly
        ...
    else:
        # Fall back to retrieval + LLM
        ...
"""

from logger import logger


def route_command(query: str, session_id: str) -> str | None:
    """
    Route special-case queries to pre-defined handlers.

    Checks the normalized query against known command patterns. If matched,
    returns a direct response string bypassing the retrieval + LLM pipeline.
    If no match, returns None so the caller can fall back to standard processing.

    Args:
        query: The user's natural language query string.
        session_id: The current conversation session ID.

    Returns:
        A direct response string if the query matches a known command,
        or None to indicate no routing match.
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
  
