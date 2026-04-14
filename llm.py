import os
import requests
from dotenv import load_dotenv

load_dotenv()


def generate_response(user_query: str, context: str, history: str = "") -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    system_prompt = """
You are DevWhisper, a senior AI software engineer assistant.

Rules:
• Explain clearly with file name and function
• Help debug errors step by step
• Use previous conversation if available
• Be conversational and voice-friendly
• Keep answers concise (4–6 sentences)

If error:
• Explain meaning
• Give cause
• Suggest fix

If code:
• Mention file + function
• Explain flow
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

Answer:
"""
            }
        ]
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body
        )

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("LLM ERROR:", e)
        print("Response:", resp.text if 'resp' in locals() else "No response")
        return "Sorry, I ran into an error while processing your request."