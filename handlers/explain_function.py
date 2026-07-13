"""
explain_function handler — voice command "explain this function"

Returns a structured explanation of one or more functions in the codebase
by combining retrieval (find them) with a strict, codebase-grounded
generation pass (explain them).

Public surface (kept intentionally small):
    INTENTS          — list of natural-language triggers this handler owns
    explain_function(query, top_k=3) -> str   — main entry point
    format_tool_call(tool_call_id, result)    — payload format used by the router

Design notes:
    - Imports retriever + llm lazily so this module is importable even
      when the FastAPI server hasn't warmed up the embedder.
    - The response format mirrors the `query_codebase` handler so the
      assistant's speech output stays consistent across commands.
    - No changes to retriever / llm / main.py — single-line registration
      in the router (see README.md § Wiring it in).
"""

from __future__ import annotations

from typing import Any


# Voice intents this handler claims. The router picks the first handler
# whose INTENTS overlap with the (lower-cased) user query.
INTENTS: tuple[str, ...] = (
    "explain this function",
    "explain that function",
    "explain function",
    "what does this function do",
    "what does that function do",
    "describe this function",
    "describe that function",
)


def matches(query: str) -> bool:
    """True iff this handler should answer the given user query."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if q in INTENTS:
        return True
    # Also accept: query CONTAINS any intent (handles "explain this function in retriever.py")
    return any(intent in q for intent in INTENTS)


def _extract_function_name(query: str) -> str | None:
    """Pull a likely function name out of the query, if present.

    Examples:
        "explain this function foo_bar"        -> "foo_bar"
        "what does parse_url do"               -> "parse_url"
        "describe function `_my_private`"     -> "_my_private"
    """
    import re

    q = query or ""
    patterns = [
        r"`([A-Za-z_][A-Za-z0-9_]*)`",                # backtick-quoted
        r"function\s+([A-Za-z_][A-Za-z0-9_]*)",      # "function foo"
        r"method\s+([A-Za-z_][A-Za-z0-9_]*)",        # "method foo"
        r"(?:does|do|did)\s+([A-Za-z_][A-Za-z0-9_]*)\s+do",  # "what does foo do"
        r"([A-Za-z_][A-Za-z0-9_]{2,})\s*\(\)?",      # trailing identifier
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None


def explain_function(query: str, top_k: int = 3) -> str:
    """Return a strict codebase-grounded explanation of the requested function(s).

    Reuses the existing retriever + llm modules so the model keeps the same
    "ONLY use provided context" rules as `query_codebase`.
    """
    # Lazy import so this module loads even if the embedder hasn't warmed up yet.
    from retriever import retrieve
    from llm import generate_response

    target = _extract_function_name(query)

    if target:
        retrieval_query = (
            f"function {target} definition signature docstring parameters return value"
        )
    else:
        retrieval_query = query or "function"

    context = retrieve(retrieval_query)
    history = ""  # explain_function is stateless — no conversation memory needed

    # Tight system override: ask the LLM to surface signature + behaviour,
    # not a generic "let me search the codebase" preamble.
    augmented_query = (
        f"{retrieval_query}\n\n"
        f"Respond ONLY with this format:\n\n"
        f"Function: <name>\n"
        f"Signature: <def line>\n"
        f"Purpose: <one short sentence grounded in the code>\n"
        f"Inputs: <param list or 'none'>\n"
        f"Returns: <type / 'None'>\n"
        f"Notes: <optional caveats / side-effects / None>"
    )

    answer = generate_response(augmented_query, context, history)
    return answer


def format_tool_call(tool_call_id: str, result: str) -> dict[str, Any]:
    """Build the JSON payload the assistant returns to Vapi."""
    return {
        "toolCallId": tool_call_id,
        "result": result,
    }