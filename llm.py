import os

from openai import OpenAI

from config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_MODEL,
    GROQ_API_KEY_ENV,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
)


def _get_client() -> OpenAI:
    """Create an OpenAI-compatible client based on the configured provider."""
    provider_api_key = os.getenv(LLM_API_KEY_ENV)
    if provider_api_key is None:
        return OpenAI(
            api_key=os.getenv(GROQ_API_KEY_ENV),
            base_url=DEFAULT_LLM_BASE_URL,
        )

    return OpenAI(
        api_key=provider_api_key or os.getenv(GROQ_API_KEY_ENV),
        base_url=os.getenv(LLM_BASE_URL_ENV, DEFAULT_LLM_BASE_URL),
    )


def _get_model() -> str:
    """Return the configured model name or the provider-specific default."""
    explicit_model = os.getenv(LLM_MODEL_ENV)
    if explicit_model:
        return explicit_model

    if os.getenv(LLM_API_KEY_ENV) is None:
        return DEFAULT_GROQ_MODEL
    return DEFAULT_OPENAI_COMPATIBLE_MODEL


def generate_response(user_query: str, context: str, history: str = "") -> str:
    system_prompt = """
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
- In <file>.py: func1, func2

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

    try:
        client = _get_client()
        model = _get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
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

INSTRUCTIONS:
- Answer strictly from code
- Do NOT add explanation unless asked
- Keep output clean and structured
""",
                },
            ],
        )

        if response.choices:
            return response.choices[0].message.content

        print("Unexpected response:", response)
        return "I could not process the response."
    except Exception as error:
        print("LLM ERROR:", error)
        return "Sorry, I ran into an error while processing your request."


def generate_response_stream(user_query: str, context: str, history: str = ""):
    system_prompt = """
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
- In <file>.py: func1, func2

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

    try:
        client = _get_client()
        model = _get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
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

INSTRUCTIONS:
- Answer strictly from code
- Do NOT add explanation unless asked
- Keep output clean and structured
""",
                },
            ],
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as error:
        print("LLM STREAM ERROR:", error)
        yield "Sorry, I ran into an error while processing your request."
