import os
from llm import _get_client, _get_model
from retriever import retrieve
from logger import logger


def can_handle(query: str) -> bool:
    """Check if the query matches the explain function, method, struct, or interface intent."""
    q = query.lower().strip()
    return any(
        phrase in q
        for phrase in (
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
    )


def handle(query: str, session_id: str) -> str:
    """Retrieve code symbol context and explain it in a voice-friendly format."""
    context = retrieve(query)
    if not context or not context.strip():
        return "I could not find any relevant functions, methods, or structs in your codebase to explain."

    system_prompt = """
You are DevWhisper, a codebase explanation assistant.
The user has asked you to explain a code entity (function, method, struct, interface, or class).
Explain the entity clearly and concisely in a voice-friendly manner (plain English, no markdown formatting like bold, italics, or bullet points).
Describe what it does, its inputs/parameters, and its outputs/return values based strictly on the provided code context across Python, JS/TS, Go, Rust, and Java.
Do not guess or assume details not present in the code.
Keep your response short (under 4 sentences).
"""


    client = _get_client()
    model = _get_model()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Explain this function based on the following code context:\n\n{context}"
                }
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.error("Error in explain_function handler", exc_info=True)
        return "Sorry, I encountered an error while trying to explain the function."
