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
)

# ---------------------------------------------------------------------------
# System prompt — strict codebase-only answering
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """
You are DevWhisper, a strict codebase analysis assistant.

STRICT RULES:
• ONLY use the provided code context
• DO NOT use general knowledge
• DO NOT explain tools or querying
• DO NOT guess
• DO NOT use phrases like "it appears", "it seems", "looks like"

IF ASKED ABOUT FUNCTIONS:
• Extract actual function names from the code
• Respond ONLY in this format:

Functions found:
- In .py: func1, func2

• If multiple files, list each file separately
• If no functions found, say:
"I could not find this in your codebase."

IF ASKED ANYTHING ELSE:
• Answer ONLY if clearly present in code
• Otherwise say:
"I could not find this in your codebase."

STYLE:
• Be direct
• No extra explanation
• Short and voice-friendly
"""

# Shared user instructions appended to every query
_USER_INSTRUCTIONS = """
INSTRUCTIONS:
- Answer strictly from code
- Do NOT add explanation unless asked
- Keep output clean and structured
"""


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


def generate_response(user_query: str, context: str, history: str = "") -> str:
    """
    Generate a complete (non-streaming) response for a user query.

    Sends the query + retrieved code context + conversation history to the
    LLM and returns the full answer string.

    Args:
        user_query: The user's natural language or code question.
        context: Retrieved code chunks from the codebase (from retriever.py).
        history: Optional conversation history string for multi-turn context.

    Returns:
        The LLM's response text, or an error message if the call fails.
    """
    try:
        client = _get_client()
        model = _get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
User question:
{user_query}

Code context:
{context}

Conversation history:
{history}

{_USER_INSTRUCTIONS}
""",
                },
            ],
        )

        if response.choices:
            # Capture and log token usage statistics if available
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                logger.info(
                    "Token Usage - Prompt: %d, Completion: %d, Total: %d",
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )

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
    response display (e.g., in the /stream endpoint).

    Args:
        user_query: The user's natural language or code question.
        context: Retrieved code chunks from the codebase.
        history: Optional conversation history string.

    Yields:
        Individual text tokens (strings) from the LLM response.
    """
    try:
        client = _get_client()
        model = _get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
User question:
{user_query}

Code context:
{context}

Conversation history:
{history}

{_USER_INSTRUCTIONS}
""",
                },
            ],
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception:
        logger.error("LLM STREAM ERROR", exc_info=True)
        yield "Sorry, I ran into an error while processing your request."
        