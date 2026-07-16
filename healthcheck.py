"""
healthcheck.py

Standalone script to verify that DevWhisper's external dependencies
(Qdrant, the embedding model, and the Groq LLM API) are reachable and
working. Useful for debugging setup issues without starting the full
FastAPI server.

Usage:
    python healthcheck.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


def check_qdrant() -> tuple[bool, str]:
    """Check that we can connect to the Qdrant cluster."""
    try:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url:
            return False, "QDRANT_URL is not set in the environment/.env file"

        client = QdrantClient(url=url, api_key=api_key)
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        return True, f"Connected. Collections found: {names or 'none'}"
    except Exception as e:
        return False, f"Could not connect to Qdrant: {e}"


def check_embedder() -> tuple[bool, str]:
    """Check that the sentence-transformers embedding model loads and works."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vector = model.encode("health check test sentence")
        if hasattr(vector, "shape"):
            dim = vector.shape[0]
        else:
            dim = len(vector)
        if dim != 384:
            return False, f"Model loaded but produced unexpected vector size: {dim}"
        return True, f"Model loaded and produced a {dim}-dim embedding"
    except Exception as e:
        return False, f"Could not load or run embedding model: {e}"


def check_llm() -> tuple[bool, str]:
    """Check that the Groq LLM API responds to a simple test prompt."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return False, "GROQ_API_KEY is not set in the environment/.env file"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": "Reply with the single word: pong"}
            ],
            "max_tokens": 5,
        }

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=15,
        )
        data = resp.json()

        if resp.status_code != 200:
            return False, f"Groq API returned status {resp.status_code}: {data}"

        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"].strip()
            return True, f"Groq API responded: '{content}'"

        return False, f"Unexpected response format: {data}"
    except Exception as e:
        return False, f"Could not reach Groq LLM API: {e}"


def run_checks():
    checks = [
        ("Qdrant connection", check_qdrant),
        ("Embedding model", check_embedder),
        ("Groq LLM API", check_llm),
    ]

    print("=" * 50)
    print("DevWhisper Dependency Health Check")
    print("=" * 50)

    results = []
    for name, fn in checks:
        print(f"\nChecking {name}...")
        passed, message = fn()
        status = "PASS" if passed else "FAIL"
        symbol = "✅" if passed else "❌"
        print(f"{symbol} {status}: {message}")
        results.append(passed)

    print("\n" + "=" * 50)
    if all(results):
        print("All dependencies are healthy. ✅")
    else:
        failed_count = results.count(False)
        print(f"{failed_count} dependency check(s) failed. ❌")
    print("=" * 50)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(run_checks())
