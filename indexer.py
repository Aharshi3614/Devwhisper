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
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    SAMPLE_CODEBASE_DIRECTORY,
    SUPPORTED_EXTENSIONS,
    BM25_INDEX_PATH
)
from logger import logger
from symbol_parser import extract_symbols_from_file

import pickle
from rank_bm25 import BM25Okapi

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
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
    "skipped": [],
    "skipped_count": 0,
}


def collect_indexable_files(directory, max_bytes=None):
    limit = MAX_FILE_SIZE_BYTES if max_bytes is None else max_bytes
    files_to_index, skipped = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, file)
            try:
                size = os.path.getsize(path)
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                skipped.append({"path": path, "size_bytes": None, "reason": "unreadable"})
                continue
            if size > limit:
                logger.warning(
                    "Skipping oversized file %s (%.2f MB exceeds %.2f MB limit)",
                    path, size / (1024 * 1024), limit / (1024 * 1024),
                )
                skipped.append({"path": path, "size_bytes": size, "reason": "oversized"})
                continue
            files_to_index.append(path)
    return files_to_index, skipped


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
                    "is_symbol": False,
                }
            )
    return chunks


def get_file_chunks(filepath: str, chunk_size: int = INDEX_CHUNK_SIZE) -> list[dict]:
    """Return all indexing chunks for *filepath*.

    For Python files this includes both AST-extracted symbols and
    traditional line-based chunks.  For other supported extensions only
    line-based chunks are returned.
    """
    chunks = []

    if filepath.lower().endswith(".py"):
        symbols = extract_symbols_from_file(filepath)
        for sym in symbols:
            chunks.append({
                "text": sym.source,
                "file": os.path.basename(filepath),
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "symbol_name": sym.name,
                "symbol_type": sym.symbol_type,
                "parent_class": sym.parent_class,
                "docstring": sym.docstring,
                "is_symbol": True,
            })

    line_chunks = chunk_file(filepath, chunk_size=chunk_size)
    chunks.extend(line_chunks)
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
                logger.warning("Corrupted .index_cache.json found.")

        if isinstance(before_cache_data, dict):
            metadata = before_cache_data.get("_metadata")
            if metadata and isinstance(metadata, dict):
                repo_version = metadata.get("embedding_version")
                if repo_version and repo_version != EMBEDDING_VERSION:
                    logger.warning(
                        f"Embedding version mismatch detected (repository version: {repo_version}, "
                        f"configured version: {EMBEDDING_VERSION}). Re-indexing is recommended."
                    )

        if "--incremental" not in sys.argv or not client.collection_exists(QDRANT_COLLECTION_NAME):
            create_collection()
        points = []
        cache_data = {}

        # Collect eligible files, skipping oversized/unreadable
        all_files, skipped_files = collect_indexable_files(directory)
        progress_state["skipped"] = skipped_files
        progress_state["skipped_count"] = len(skipped_files)
        total_files = len(all_files)
        progress_state["total"] = total_files
        skip_msg = f" ({len(skipped_files)} skipped)" if skipped_files else ""
        progress_state["message"] = f"Found {total_files} file(s) to index{skip_msg}"

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

            chunks = get_file_chunks(path)
            line_count = sum(1 for c in chunks if not c.get("is_symbol"))
            sym_count = sum(1 for c in chunks if c.get("is_symbol"))
            msg = f" {file} → {line_count} line chunks"
            if sym_count:
                msg += f", {sym_count} symbols"
            print(msg)

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

        all_chunks = []
        for path in all_files:
            chunks = get_file_chunks(path)
            all_chunks.extend(chunks)

        if all_chunks:
            tokenize_corpus = [tokenize(c["text"]) for c in all_chunks]
            bm25 = BM25Okapi(tokenize_corpus)
            with open(BM25_INDEX_PATH,"wb") as f:
                pickle.dump({
                    "bm25": bm25,
                    "corpus": [c["text"] for c in all_chunks],
                    "chunks": all_chunks
                }, f)
            print(f"BM25 index saved ({len(all_chunks)} chunks) to {BM25_INDEX_PATH}")

        cache_data["_metadata"] = {
            "repository_name": os.path.basename(os.path.abspath(directory)),
            "indexing_timestamp": datetime.now(timezone.utc).isoformat(),
            "indexed_file_count": len(cache_data),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_version": EMBEDDING_VERSION,
            "skipped_files": skipped_files,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
        }

        with open(".index_cache.json", "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        skip_summary = f", {len(skipped_files)} file(s) skipped" if skipped_files else ""
        progress_state.update({
            "running": False,
            "percent": 100,
            "status": "done",
            "message": f"Indexing complete. {total_files} file(s) processed{skip_summary}, {len(points)} chunks uploaded.",
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

def tokenize(text:str) -> list[str]:
    """Split code text into lowercased word tokens for BM25."""
    import re
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]

if __name__ == "__main__":
    index_directory(SAMPLE_CODEBASE_DIRECTORY)