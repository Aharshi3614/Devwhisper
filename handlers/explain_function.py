
from llm import _get_client, _get_model
from retriever import retrieve
from logger import logger


def can_handle(query: str) -> bool:
    """Check whether the query requests an explanation of a code entity."""

    if not isinstance(query, str) or not query.strip():
        return False

    q = query.lower().strip()

    phrases = (
        "explain this function",
        "explain the function",
        "explain function",
        "explain this method",
        "explain the method",
        "explain method",
        "explain this struct",
        "explain struct",
        "explain this interface",
        "explain interface",
        "explain this class",
        "explain class",
    )

    return any(phrase in q for phrase in phrases)


def handle(query: str, session_id: str) -> str:
    """Retrieve code context and generate a voice-friendly explanation."""

    if not isinstance(query, str) or not query.strip():
        return "Please provide a valid query to explain."

    if not isinstance(session_id, str) or not session_id.strip():
        return "A valid session ID is required."

    try:
        context = retrieve(query)

        if not isinstance(context, str) or not context.strip():
            return (
                "I could not find any relevant functions, methods, "
                "structs, interfaces, or classes in your codebase to explain."
            )

        system_prompt = """
You are DevWhisper, a codebase explanation assistant.

The user wants an explanation of a code entity such as a function,
method, struct, interface, or class.

Explain clearly and concisely in a voice-friendly format.
Use plain English without Markdown formatting, bullet points,
or unnecessary technical jargon.

Describe:
1. What the entity does.
2. Its inputs and parameters.
3. Its outputs or return values.

Base your explanation strictly on the provided code context.
Support Python, JavaScript, TypeScript, Go, Rust, and Java.
Do not guess or assume details that are not present in the code.

Keep the explanation under four sentences.
"""

        client = _get_client()
        model = _get_model()

        if client is None or not model:
            logger.error("LLM client or model is not configured.")
            return "Sorry, the explanation service is currently unavailable."

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                },
                {
                    "role": "user",
                    "content": (
                        f"Explain the requested code entity based on "
                        f"the following code context:\n\n{context}"
                    ),
                },
            ],
            temperature=0.2,
        )

        if not response or not response.choices:
            logger.error("LLM returned an empty response.")
            return "Sorry, I could not generate an explanation."

        message = response.choices[0].message
        answer = message.content if message else None

        if not isinstance(answer, str) or not answer.strip():
            logger.error("LLM returned empty explanation content.")
            return "Sorry, I could not generate an explanation."

        return answer.strip()

    except Exception:
        logger.error(
            "Error in explain_function handler",
            exc_info=True,
        )
        return (
            "Sorry, I encountered an error while trying "
            "to explain the code entity."
        )
