import hashlib
import os
import uuid
import sys
import json
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    INDEX_CHUNK_OVERLAP,
    INDEX_CHUNK_SIZE,
    QDRANT_API_KEY_ENV,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL_ENV,
    SAMPLE_CODEBASE_DIRECTORY,
    SUPPORTED_EXTENSIONS,
)

client = QdrantClient(
    url=os.getenv(QDRANT_URL_ENV),
    api_key=os.getenv(QDRANT_API_KEY_ENV),
)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Shared progress state — written by index_directory(), read by /index/progress
progress_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "percent": 0,
    "current_file": "",
    "status": "idle",  # idle | running | done | error
    "message": "",
}


def create_collection():
    try:
        client.delete_collection(QDRANT_COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=Distance.COSINE,
        ),
    )


def chunk_file(filepath, chunk_size=INDEX_CHUNK_SIZE):
    if chunk_size <= INDEX_CHUNK_OVERLAP:
        raise ValueError("chunk_size must be greater than INDEX_CHUNK_OVERLAP")

    with open(filepath, "r", errors="ignore") as file_handle:
        lines = file_handle.readlines()

    chunks = []
    step = chunk_size - INDEX_CHUNK_OVERLAP
    for index in range(0, len(lines), step):
        chunk = "".join(lines[index:index + chunk_size])
        if chunk.strip():
            chunks.append(
                {
                    "text": chunk,
                    "file": os.path.basename(filepath),
                    "start_line": index + 1,
                }
            )
    return chunks


def index_directory(directory):
    progress_state.update({"running": True, "current": 0, "total": 0,
                           "percent": 0, "current_file": "", "status": "running", "message": "Starting..."})
    try:
        before_cache_data = {}
        if os.path.exists(".index_cache.json"):
            try:
                with open(".index_cache.json", "r", encoding="utf-8") as f:
                    before_cache_data = json.load(f)
            except Exception:
                print("Warning: Corrupted .index_cache.json found.")

        if isinstance(before_cache_data, dict):
            metadata = before_cache_data.get("_metadata")
            if metadata and isinstance(metadata, dict):
                repo_version = metadata.get("embedding_version")
                if repo_version and repo_version != EMBEDDING_VERSION:
                    print(
                        f"Warning: Embedding version mismatch detected (repository version: {repo_version}, "
                        f"configured version: {EMBEDDING_VERSION}). Re-indexing is recommended."
                    )

        if "--incremental" not in sys.argv or not client.collection_exists(QDRANT_COLLECTION_NAME):
            create_collection()
        points = []
        cache_data = {}

        # Collect all eligible files first so we know the total
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        total_files = len(all_files)
        progress_state["total"] = total_files
        progress_state["message"] = f"Found {total_files} file(s) to index"

        for idx, path in enumerate(all_files, start=1):
            file = os.path.basename(path)
            progress_state.update({
                "current": idx,
                "percent": round((idx / total_files) * 100) if total_files else 100,
                "current_file": file,
                "message": f"Indexing {file} ({idx}/{total_files})",
            })

            cache_data[path] = {
                "mtime": os.path.getmtime(path),
                "hash": get_file_hash(path)
            }

            if "--incremental" in sys.argv and path in before_cache_data:
                if abs(before_cache_data[path]["mtime"] - cache_data[path]["mtime"]) <= 0.001:
                    continue
                else:
                    if before_cache_data[path]["hash"] == cache_data[path]["hash"]:
                        continue

            chunks = chunk_file(path)
            print(f" {file} → {len(chunks)} chunks")

            for chunk in chunks:
                vector = embedder.encode(chunk["text"]).tolist()
                unique_str = f"{path}_{chunk['start_line']}"
                stable_id = str(uuid.uuid5(uuid.NAMESPACE_OID, unique_str))
                points.append(
                    PointStruct(
                        id=stable_id,
                        vector=vector,
                        payload=chunk,
                    )
                )

        if points:
            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points,
            )
            print(f"\nDone. Indexed {len(points)} total chunks into Qdrant.")
        else:
            print("\nNo changes detected. Nothing to upsert.")

        cache_data["_metadata"] = {
            "repository_name": os.path.basename(os.path.abspath(directory)),
            "indexing_timestamp": datetime.now(timezone.utc).isoformat(),
            "indexed_file_count": len(cache_data),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_version": EMBEDDING_VERSION
        }

        with open(".index_cache.json", "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        progress_state.update({
            "running": False,
            "percent": 100,
            "status": "done",
            "message": f"Indexing complete. {total_files} file(s) processed, {len(points)} chunks uploaded.",
        })

    except Exception as e:
        progress_state.update({
            "running": False,
            "status": "error",
            "message": f"Indexing failed: {e}",
        })
        raise

def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    index_directory(SAMPLE_CODEBASE_DIRECTORY)