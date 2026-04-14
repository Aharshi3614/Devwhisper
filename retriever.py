import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, top_k: int = 6) -> str:
    vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name="devwhisper",
        query=vector,
        limit=top_k
    ).points
    context = ""
    for r in results:
        context += f"\n# File: {r.payload['file']}, line {r.payload['start_line']}\n"
        context += r.payload["text"] + "\n"
    return context