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
You are DevWhisper, a strict AI code assistant.

Rules:
• ONLY answer using the provided code context
• DO NOT guess or invent anything
• If answer is not clearly in the code, say:
  "I could not find this in your codebase."

For code questions:
• List actual functions found
• Mention file names
• Be precise and factual

For errors:
• Explain meaning, cause, and fix ONLY if relevant code is present

Keep answers short and voice-friendly.
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

Code context:
{context}

Instructions:
- Answer ONLY using the code above
- If not found, say you could not find it
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

        # 🔥 Safe response handling
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            print("Unexpected response:", data)
            return "I could not process the response properly."

    except Exception as e:
        print("LLM ERROR:", e)
        print("Response:", resp.text if 'resp' in locals() else "No response")
        return "Sorry, I ran into an error while processing your request."