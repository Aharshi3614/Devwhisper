from .explain_function import can_handle as can_explain, handle as handle_explain


def route_command(query: str, session_id: str) -> str | None:
    """Route the query to custom voice command handlers if matched.

    Returns the response string if a handler matches, or None otherwise.
    """
    if can_explain(query):
        return handle_explain(query, session_id)
    return None
