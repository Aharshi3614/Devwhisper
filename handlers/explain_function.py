import os
from llm import _get_client, _get_model
from retriever import retrieve


def can_handle(query: str) -> bool:
    """Check if the query matches the explain function intent."""
    q = query.lower().strip()
    return "explain this function" in q or "explain the function" in q or "explain function" in q


def handle(query: str, session_id: str) -> str:
    """Retrieve function context and explain it in a voice-friendly format."""
    context = retrieve(query)
    if not context or not context.strip():
        return "I could not find any relevant functions in your codebase to explain."

    system_prompt = """
You are DevWhisper, a codebase explanation assistant.
The user has asked you to explain a function.
Explain the function clearly and concisely in a voice-friendly manner (plain English, no markdown formatting like bold, italics, or bullet points).
Describe what the function does, its inputs, and its outputs based strictly on the provided code context.
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
        print(f"Error in explain_function handler: {e}")
        return "Sorry, I encountered an error while trying to explain the function."
