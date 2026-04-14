import os, requests
from dotenv import load_dotenv
load_dotenv()

def generate_response(user_query: str, context: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DevWhisper, a voice-native AI pair programmer. "
                    "You have been given code chunks from the developer's actual project. "
                    "Search through ALL the provided code carefully and answer specifically. "
                    "If the answer is in the code, state exactly which file and function it is in. "
                    "Respond in plain spoken English, no markdown, under 4 sentences."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Developer asked: {user_query}\n\n"
                    f"Here are ALL the relevant code chunks from their project:\n{context}\n\n"
                    "Answer specifically based on the code above."
                )
            }
        ]
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body
    )
    return resp.json()["choices"][0]["message"]["content"]