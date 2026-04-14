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
• DO NOT talk about tools or querying
• DO NOT give generic explanations
• DO NOT guess

IF ASKED ABOUT FUNCTIONS:
• Extract actual function names from the code
Format response clearly:
Functions found:
- In model.py: train_model, evaluate_model, save_model
- In pipeline.py: run_pipeline

Speak file names naturally (say dot py, not letter by letter).
• If none found → say "I could not find this in your codebase."

IF ASKED ANYTHING ELSE:
• Answer ONLY if clearly present in code
• Otherwise say not found

Be direct. No extra talk. No explanations unless asked.
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
- Extract answer strictly from code
- Do NOT explain how to query
- Do NOT give general knowledge
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
            return "I could not process the response."

    except Exception as e:
        print("LLM ERROR:", e)
        return "Sorry, I ran into an error."