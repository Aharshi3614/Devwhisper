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
You are DevWhisper, a strict codebase-aware AI assistant.

CRITICAL RULES:
• ONLY answer using the provided code context
• DO NOT use general knowledge
• DO NOT give generic explanations
• If the answer is not explicitly in the code, say:
  "I could not find this in your codebase."

WHEN USER ASKS "WHY" OR "HOW":
• Look for actual issues in the code
• If no issue is found → say not found
• DO NOT guess

FOR FUNCTION QUESTIONS:
• List real functions from context
• Mention file names

FOR DEBUGGING:
• Only explain errors if relevant code is present

Keep answers short, precise, and voice-friendly.
"""

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Previous conversation:
{history}

User question:
{user_query}

Code context:
{context}

STRICT INSTRUCTIONS:
- Answer ONLY using the code above
- Do NOT use general knowledge
- If not found, say "I could not find this in your codebase"
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

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            print("Unexpected response:", data)
            return "I could not process the response properly."

    except Exception as e:
        print("LLM ERROR:", e)
        print("Response:", resp.text if 'resp' in locals() else "No response")
        return "Sorry, I ran into an error while processing your request."