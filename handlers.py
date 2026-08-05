"""
handlers.py — Command routing, request processing pipeline, and webhook event handlers for DevWhisper.

This module provides command routing, the core request processing pipeline, 
and modular webhook event handlers to decouple Vapi payload processing from main.py.
"""

from logger import logger
from cache import get as cache_get, put as cache_put
from retriever import retrieve
from llm import generate_response
from fastapi.responses import JSONResponse
from errors import error_response
import json


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
    """
    cached = cache_get(query)
    if cached is not None:
        memory_updater(session_id, query, cached)
        logger.info("Cache hit for query: %s", query)
        return cached, []

    context, sources = retrieve(query, include_sources=True, repositories=repositories)
    history = memory_getter(session_id)

    answer = route_command(query, session_id)
    if not answer:
        answer = generate_response(query, context, history)

    if answer and answer.strip() and sources:
        answer += "\n\n**Sources used:** " + ", ".join(f"`{s}`" for s in sources)

    if answer and answer.strip():
        cache_put(query, answer)
    
    memory_updater(session_id, query, answer)

    return answer, sources


# ---------------------------------------------------------------------------
# Dedicated Webhook Event Handlers (Issue #188)
# ---------------------------------------------------------------------------

def handle_assistant_request() -> JSONResponse:
    """Handle Vapi 'assistant-request' event initialization."""
    return JSONResponse({
        "assistant": {
            "firstMessage": "Hey, DevWhisper here. What are you building or debugging?",
            "model": {
                "provider": "openai",
                "model": "gpt-4o",
                "functions": [{
                    "name": "query_codebase",
                    "description": "Search and explain code or debug errors",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }]
            },
            "voice": {"provider": "11labs", "voiceId": "burt"}
        }
    })


def handle_tool_calls(message: dict, session_id: str, memory_getter, memory_updater) -> JSONResponse:
    """Handle Vapi 'function-call' and 'tool-calls' execution events."""
    msg_type = message.get("type", "")
    tools = []

    if msg_type == "function-call":
        tools = [{
            "id": "single",
            "function": message.get("functionCall", {})
        }]
    else:
        tools = message.get("toolCalls", [])

    results = []

    for tool in tools:
        fn = tool.get("function", {})
        fn_name = fn.get("name", "")

        params = fn.get("arguments") or fn.get("parameters") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse command parameters: %s", e)
                return error_response(400, "Invalid JSON in command parameters. Try rephrasing.")

        if fn_name == "query_codebase":
            query = params.get("query", "")
            if not query:
                return error_response(400, "Query parameter is required and cannot be empty.")

            answer, _ = process_query_pipeline(
                query=query,
                session_id=session_id,
                memory_getter=memory_getter,
                memory_updater=memory_updater
            )

            results.append({
                "toolCallId": tool.get("id", "single"),
                "result": answer
            })

    return JSONResponse({"results": results})
