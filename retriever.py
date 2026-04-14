import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, top_k: int = 6) -> str:
    vector = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name="devwhisper",
        query=vector,
        limit=top_k
    ).points

    structured_context = []

    for i, r in enumerate(results):
        payload = r.payload

        file = payload.get("file", "unknown")
        line = payload.get("start_line", "?")
        code = payload.get("text", "")

        # Extract function name
        function_name = "unknown"
        for line_text in code.split("\n"):
            if line_text.strip().startswith("def "):
                function_name = line_text.strip().split("(")[0].replace("def ", "")
                break

        structured_context.append(
            f"""
Result {i+1}:
File: {file}
Function: {function_name}
Start Line: {line}

Code:
{code}
"""
        )

    return "\n\n".join(structured_context)