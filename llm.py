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
User question:
{user_query}

Code context:
{context}

INSTRUCTIONS:
- Answer strictly from code
- Do NOT add explanation unless asked
- Keep output clean and structured
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
            return "I could not process the response."

    except Exception as e:
        print("LLM ERROR:", e)
        return "Sorry, I ran into an error while processing your request."