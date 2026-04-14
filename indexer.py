import os, uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "devwhisper"

def create_collection():
    try:
        client.delete_collection(COLLECTION)
    except:
        pass
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def chunk_file(filepath, chunk_size=15):
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()
    chunks = []
    for i in range(0, len(lines), chunk_size - 3):
        chunk = "".join(lines[i:i+chunk_size])
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "file": os.path.basename(filepath),
                "start_line": i + 1
            })
    return chunks

def index_directory(directory):
    create_collection()
    points = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                chunks = chunk_file(path)
                print(f"  {file} → {len(chunks)} chunks")
                for chunk in chunks:
                    vector = embedder.encode(chunk["text"]).tolist()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=chunk
                    ))
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\nDone. Indexed {len(points)} total chunks into Qdrant.")

if __name__ == "__main__":
    index_directory("./sample_codebase")