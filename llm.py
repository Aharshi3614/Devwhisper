import os, requests
from dotenv import load_dotenv

load_dotenv()

def generate_response(user_query: str, context: str, history: str = "") -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    system_prompt = """
You are DevWhisper, a senior AI software engineer assistant.

Your job:
• Explain code clearly with file name and function
• Help debug errors step by step
• Use previous conversation context if available
• Be conversational and natural (voice-friendly)
• Keep answers concise but informative (4–6 sentences max)

If error is given:
• Explain what it means
• Give likely causes
• Suggest fixes

If code question:
• Mention file + function
• Explain how it works
"""

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Previous conversation:
{history}

User question:
{user_query}

Relevant code:
{context}

Answer clearly:
"""
            }
        ]
    }

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body
    )

    return resp.json()["choices"][0]["message"]["content"]