"""
llm.py — LLM integration layer for DevWhisper.

This module handles communication with the configured LLM provider (Groq by
default, or any OpenAI-compatible API). It provides two response modes:

    1. generate_response()      — Synchronous, returns the complete answer string.
    2. generate_response_stream() — Streaming, yields tokens as they arrive.

Both functions inject a strict system prompt that constrains the model to
answer ONLY from the provided code context, avoiding hallucination and
general-knowledge answers.

Configuration:
    - GROQ_API_KEY: Default provider API key.
    - LLM_API_KEY / LLM_BASE_URL / LLM_MODEL: Override for custom OpenAI-compatible providers.

Dependencies:
    - openai (OpenAI-compatible client library)
"""

from openai import OpenAI
from logger import logger

from config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_MODEL,
    GROQ_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_PROMPT_CONTEXT_TOKENS,
    ENABLE_CONTEXT_COMPRESSION,
)
from context_packer import pack_context

# ---------------------------------------------------------------------------
# System prompt — strict codebase-only answering
# ---------------------------------------------------------------------------
from prompt_builder import build_messages, SYSTEM_PROMPT, USER_INSTRUCTIONS, prepare_context, prepare_user_content


# Re-export for backwards compatibility
_SYSTEM_PROMPT = SYSTEM_PROMPT
_USER_INSTRUCTIONS = USER_INSTRUCTIONS


def _get_client() -> OpenAI:
    """
    Create an OpenAI-compatible client based on the configured provider.

    Priority:
        1. If LLM_API_KEY is set → use custom provider (LLM_BASE_URL + LLM_API_KEY).
        2. Otherwise → use Groq (DEFAULT_LLM_BASE_URL + GROQ_API_KEY).

    Returns:
        Configured OpenAI client instance.
    """
    if LLM_API_KEY is None:
        return OpenAI(
            api_key=GROQ_API_KEY,
            base_url=DEFAULT_LLM_BASE_URL,
        )

    return OpenAI(
        api_key=LLM_API_KEY or GROQ_API_KEY,
        base_url=LLM_BASE_URL,
    )


def _get_model() -> str:
    """
    Return the configured model name or the provider-specific default.

    Priority:
        1. If LLM_MODEL is set → return it.
        2. If using Groq (LLM_API_KEY is None) → return DEFAULT_GROQ_MODEL.
        3. Otherwise → return DEFAULT_OPENAI_COMPATIBLE_MODEL.

    Returns:
        Model identifier string for the chat.completions.create() call.
    """
    if LLM_MODEL:
        return LLM_MODEL

    if LLM_API_KEY is None:
        return DEFAULT_GROQ_MODEL
    return DEFAULT_OPENAI_COMPATIBLE_MODEL


import time
from typing import Dict, Any, Generator

_llm_telemetry_records: list[Dict[str, Any]] = []

def get_llm_provider_info() -> Dict[str, Any]:
    """Return active LLM provider configuration, active model, and telemetry stats."""
    provider = "custom" if LLM_API_KEY else "groq"
    model = _get_model()
    base_url = LLM_BASE_URL if LLM_API_KEY else DEFAULT_LLM_BASE_URL
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "is_custom_provider": bool(LLM_API_KEY),
        "total_requests": len(_llm_telemetry_records),
        "recent_telemetry": _llm_telemetry_records[-10:],
    }

def record_telemetry(data: Dict[str, Any]) -> None:
    """Store recent execution telemetry metrics."""
    _llm_telemetry_records.append(data)
    if len(_llm_telemetry_records) > 50:
        del _llm_telemetry_records[:-50]

from request_context import RequestContext

def generate_response(
    user_query: str | RequestContext = "",
    context: str = "",
    history: str = "",
    req_context: RequestContext | None = None,
) -> str:
    """
    Generate a complete (non-streaming) response for a user query.
    Supports RequestContext object passed via req_context or as first parameter.
    """
    if isinstance(user_query, RequestContext):
        req_context = user_query
        user_query = req_context.user_query
    elif req_context is not None and not user_query:
        user_query = req_context.user_query
    
    start_time = time.time()
    try:
        client = _get_client()
        model = _get_model()
        packed_context, pack_stats = pack_context(
            context,
            max_tokens=MAX_PROMPT_CONTEXT_TOKENS,
            enable_compression=ENABLE_CONTEXT_COMPRESSION,
        )
        messages = build_messages(user_query, packed_context, history)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)

        if response.choices:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or 0
                logger.info(
                    "Token Usage - Prompt: %d, Completion: %d, Total: %d",
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )

            record_telemetry({
                "type": "sync",
                "model": model,
                "duration_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "context_stats": pack_stats,
                "timestamp": time.time(),
            })

            return response.choices[0].message.content

        logger.error("Unexpected response: %s", response)
        return "I could not process the response."
    except Exception:
        logger.error("LLM ERROR", exc_info=True)
        return "Sorry, I ran into an error while processing your request."


def generate_response_stream(user_query: str, context: str, history: str = ""):
    """
    Generate a streaming response for a user query.

    Yields tokens as they are received from the LLM, enabling real-time
    response display (e.g., in the /stream endpoint). Tracks token count and latency.

    Args:
        user_query: The user's natural language or code question.
        context: Retrieved code chunks from the codebase.
        history: Optional conversation history string.

    Yields:
        Individual text tokens (strings) from the LLM response.
    """
    start_time = time.time()
    token_count = 0
    model = _get_model()
    try:
        client = _get_client()
        packed_context, _ = pack_context(
            context,
            max_tokens=MAX_PROMPT_CONTEXT_TOKENS,
            enable_compression=ENABLE_CONTEXT_COMPRESSION,
        )
        messages = build_messages(user_query, packed_context, history)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )


        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token_chunk = chunk.choices[0].delta.content
                token_count += 1
                yield token_chunk

        duration_ms = round((time.time() - start_time) * 1000, 2)
        record_telemetry({
            "type": "stream",
            "model": model,
            "duration_ms": duration_ms,
            "token_chunks": token_count,
            "timestamp": time.time(),
        })

    except Exception:
        logger.error("LLM STREAM ERROR", exc_info=True)
        yield "Sorry, I ran into an error while processing your request."
        