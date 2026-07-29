import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any
import json
from datetime import datetime, timezone

from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    INDEX_CHUNK_OVERLAP,
    INDEX_CHUNK_SIZE,
    QDRANT_COLLECTION_NAME,
    SAMPLE_CODEBASE_DIRECTORY,
    SUPPORTED_EXTENSIONS,
    BM25_INDEX_PATH
)
from dependencies import IndexingDependencies, get_indexing_dependencies


@dataclass
class IndexValidationReport:
    """Summary of index integrity checks for the current codebase."""

    expected_chunk_count: int = 0
    indexed_chunk_count: int = 0
    missing_point_ids: list[str] = field(default_factory=list)
    unexpected_point_ids: list[str] = field(default_factory=list)
    metadata_issues: list[str] = field(default_factory=list)
    file_issues: list[str] = field(default_factory=list)
    collection_issues: list[str] = field(default_factory=list)
import pickle
from rank_bm25 import BM25Okapi

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


    @property
    def is_valid(self) -> bool:
        return not (
            self.missing_point_ids
            or self.unexpected_point_ids
            or self.metadata_issues
            or self.file_issues
            or self.collection_issues
        )


progress_state = {
    "status": "idle",
    "current_file": None,
    "files_processed": 0,
    "chunks_indexed": 0,
    "last_validation": None,
}


def _safe_load_json(path: str) -> dict[str, Any]:
    """Load a JSON file but treat missing or broken caches as empty."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read {path}; rebuilding cache from disk.")
        return {}


def _stable_point_id(path: str, start_line: int) -> str:
    """Generate a stable identifier for a file chunk."""
    unique_str = f"{path}_{start_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, unique_str))


def _text_hash(text: str) -> str:
    """Compute a deterministic content hash for chunk payload validation."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _file_hash(filepath: str) -> str:
    """Compute a file hash for stale-index detection."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _build_chunk_payload(
    path: str,
    chunk: dict[str, Any],
    chunk_index: int,
    chunk_count: int,
    file_hash: str,
) -> dict[str, Any]:
    """Attach metadata needed for validation to each indexed chunk."""
    text = chunk["text"]
    payload = dict(chunk)
    payload.update(
        {
            "source_path": path,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "chunk_hash": _text_hash(text),
            "file_hash": file_hash,
            "end_line": chunk["start_line"] + max(len(text.splitlines()) - 1, 0),
        }
    )
    return payload


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


def _collect_expected_index_state(
    directory: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Build the expected point map for the current filesystem state."""
    expected_points: dict[str, dict[str, Any]] = {}
    per_file_counts: dict[str, int] = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                path = os.path.join(root, file)
                chunks = chunk_file(path)
                file_hash = _file_hash(path)
                per_file_counts[path] = len(chunks)
                for chunk_index, chunk in enumerate(chunks):
                    point_id = _stable_point_id(path, chunk["start_line"])
                    expected_points[point_id] = {
                        "path": path,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "payload": _build_chunk_payload(
                            path,
                            chunk,
                            chunk_index,
                            len(chunks),
                            file_hash,
                        ),
                    }

    return expected_points, per_file_counts


def _scroll_collection_points(client, collection_name: str) -> list[Any]:
    """Fetch all points from a Qdrant collection using pagination."""
    points: list[Any] = []
    offset = None

    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            break

    return points


def validate_index(
    directory: str,
    dependencies: IndexingDependencies | None = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> IndexValidationReport:
    """Validate that the vector index matches the current codebase state."""
    report = IndexValidationReport()
    resolved = dependencies or get_indexing_dependencies()

    if not resolved.client.collection_exists(collection_name):
        report.collection_issues.append(f"Collection does not exist: {collection_name}")
        return report

    expected_points, per_file_counts = _collect_expected_index_state(directory)
    report.expected_chunk_count = len(expected_points)

    try:
        actual_points = _scroll_collection_points(resolved.client, collection_name)
    except Exception as exc:
        report.collection_issues.append(f"Could not read collection '{collection_name}': {exc}")
        return report

    report.indexed_chunk_count = len(actual_points)
    actual_by_id = {str(point.id): point for point in actual_points}

    expected_ids = set(expected_points)
    actual_ids = set(actual_by_id)
    report.missing_point_ids.extend(sorted(expected_ids - actual_ids))
    report.unexpected_point_ids.extend(sorted(actual_ids - expected_ids))

    for point_id, point in actual_by_id.items():
        payload = point.payload or {}
        if not isinstance(payload, dict):
            report.metadata_issues.append(f"{point_id}: payload is not a dictionary")
            continue

        required_fields = (
            "text",
            "file",
            "start_line",
            "source_path",
            "chunk_index",
            "chunk_count",
            "chunk_hash",
            "file_hash",
        )

        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            report.metadata_issues.append(
                f"{point_id}: missing metadata fields {', '.join(missing_fields)}"
            )
            continue

        text = payload.get("text")
        file_name = payload.get("file")
        source_path = payload.get("source_path")
        start_line = payload.get("start_line")
        chunk_index = payload.get("chunk_index")
        chunk_count = payload.get("chunk_count")
        chunk_hash = payload.get("chunk_hash")
        file_hash = payload.get("file_hash")

        if not isinstance(text, str) or not text.strip():
            report.metadata_issues.append(f"{point_id}: chunk text is empty or invalid")
        if not isinstance(file_name, str) or not file_name.strip():
            report.metadata_issues.append(f"{point_id}: file name is missing or invalid")
        if not isinstance(source_path, str) or not source_path.strip():
            report.metadata_issues.append(f"{point_id}: source_path is missing or invalid")
        elif not os.path.exists(source_path):
            report.file_issues.append(f"{point_id}: source file is missing on disk: {source_path}")
        if not isinstance(start_line, int) or start_line < 1:
            report.metadata_issues.append(f"{point_id}: start_line must be a positive integer")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            report.metadata_issues.append(f"{point_id}: chunk_index must be a non-negative integer")
        if not isinstance(chunk_count, int) or chunk_count < 1:
            report.metadata_issues.append(f"{point_id}: chunk_count must be a positive integer")
        if not isinstance(chunk_hash, str) or chunk_hash != _text_hash(text or ""):
            report.metadata_issues.append(f"{point_id}: chunk_hash does not match chunk text")
        if not isinstance(file_hash, str):
            report.metadata_issues.append(f"{point_id}: file_hash is missing or invalid")
        elif isinstance(source_path, str) and os.path.exists(source_path):
            current_file_hash = _file_hash(source_path)
            if current_file_hash != file_hash:
                report.file_issues.append(
                    f"{point_id}: file hash differs from the current source file"
                )

        if isinstance(source_path, str) and isinstance(chunk_count, int):
            expected_count = per_file_counts.get(source_path)
            if expected_count is not None and expected_count != chunk_count:
                report.file_issues.append(
                    f"{point_id}: chunk_count={chunk_count} but filesystem now has {expected_count} chunks for {source_path}"
                )

    return report


def print_validation_report(report: IndexValidationReport) -> None:
    """Render a human-readable integrity report to stdout."""
    print("\nIndex validation report")
    print("-" * 24)
    print(f"Expected chunks: {report.expected_chunk_count}")
    print(f"Indexed chunks: {report.indexed_chunk_count}")
    print(f"Missing chunks: {len(report.missing_point_ids)}")
    print(f"Unexpected chunks: {len(report.unexpected_point_ids)}")
    print(f"Metadata issues: {len(report.metadata_issues)}")
    print(f"File issues: {len(report.file_issues)}")
    print(f"Collection issues: {len(report.collection_issues)}")

    if report.collection_issues:
        print("\nCollection issues:")
        for issue in report.collection_issues:
            print(f"  - {issue}")

    if report.metadata_issues:
        print("\nMetadata issues:")
        for issue in report.metadata_issues:
            print(f"  - {issue}")

    if report.file_issues:
        print("\nFile issues:")
        for issue in report.file_issues:
            print(f"  - {issue}")

    if report.missing_point_ids:
        print("\nMissing point IDs:")
        for point_id in report.missing_point_ids[:20]:
            print(f"  - {point_id}")
        if len(report.missing_point_ids) > 20:
            print(f"  - ... and {len(report.missing_point_ids) - 20} more")

    if report.unexpected_point_ids:
        print("\nUnexpected point IDs:")
        for point_id in report.unexpected_point_ids[:20]:
            print(f"  - {point_id}")
        if len(report.unexpected_point_ids) > 20:
            print(f"  - ... and {len(report.unexpected_point_ids) - 20} more")

    if report.is_valid:
        print("\nIndex validation passed.")
    else:
        print("\nIndex validation found inconsistencies.")


def create_collection(client):
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


def index_directory(
    directory,
    dependencies: IndexingDependencies | None = None,
):
    resolved = dependencies or get_indexing_dependencies()
    progress_state.update(
        {
            "status": "indexing",
            "current_file": None,
            "files_processed": 0,
            "chunks_indexed": 0,
            "last_validation": None,
        }
    )
    before_cache_data = _safe_load_json(".index_cache.json")
    if "--incremental" not in sys.argv or not resolved.client.collection_exists(QDRANT_COLLECTION_NAME):
        create_collection(resolved.client)

    points = []
    cache_data = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                path = os.path.join(root, file)
                progress_state["current_file"] = path
                file_hash = get_file_hash(path)
                cache_data[path] = {
                    "mtime": os.path.getmtime(path),
                    "hash": file_hash,
                }
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
                    if before_cache_data[path]["hash"] == cache_data[path]["hash"]:
                        continue

                chunks = chunk_file(path)
                print(f" {file} -> {len(chunks)} chunks")

                for chunk_index, chunk in enumerate(chunks):
                    vector = resolved.embedder.encode(chunk["text"]).tolist()
                    stable_id = _stable_point_id(path, chunk["start_line"])
                    payload = _build_chunk_payload(
                        path,
                        chunk,
                        chunk_index,
                        len(chunks),
                        file_hash,
                    )
                    points.append(
                        PointStruct(
                            id=stable_id,
                            vector=vector,
                            payload=payload,
                        )
                    )
                progress_state["files_processed"] += 1
                progress_state["chunks_indexed"] += len(chunks)

    if points:
        resolved.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )
        print(f"\nDone. Indexed {len(points)} total chunks into Qdrant.")
    else:
        print("\nNo changes detected. Nothing to upsert.")

    with open(".index_cache.json", "w", encoding="utf-8") as handle:
        json.dump(cache_data, handle, indent=2, ensure_ascii=False)

    validation_report = validate_index(directory, dependencies=resolved)
    progress_state["status"] = "validated" if validation_report.is_valid else "validation_failed"
    progress_state["last_validation"] = validation_report
    print_validation_report(validation_report)
    return validation_report


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

        all_chunks = []
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                    path = os.path.join(root, file)
                    chunks = chunk_file(path)
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
    return _file_hash(filepath)

def tokenize(text:str) -> list[str]:
    """Split code text into lowercased word tokens for BM25."""
    import re
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]

if __name__ == "__main__":
    index_directory(SAMPLE_CODEBASE_DIRECTORY)
