"""indexer.py — Codebase indexing pipeline for DevWhisper.

This module handles the ingestion, chunking, embedding, and storage of source
files into Qdrant (vector DB) and a local BM25 keyword index. It supports
both full and incremental re-indexing, file-size limits, and progress tracking
for real-time monitoring via SSE.

Files listed in the codebase's `.gitignore` files (root or nested) are
automatically skipped during collection.

Key components:
  - collect_indexable_files(): Walks the codebase directory and filters eligible files.
  - load_gitignore_rules(): Collects root + nested .gitignore rules into matchers.
  - create_collection(): (Re)creates the Qdrant vector collection.
  - chunk_file(): Splits a file into overlapping line-based chunks.
  - get_file_chunks(): Returns symbol chunks + line chunks for Python files.
  - index_directory(): Main indexing orchestrator — full pipeline.
  - get_file_hash(): Computes MD5 hash for incremental change detection.
  - tokenize(): Simple word tokenizer for BM25.

Progress tracking:
  The global `progress_state` dict is updated throughout indexing and
  consumed by the /index/progress SSE endpoint in main.py.

Usage:
  python indexer.py              # Full re-index
  python indexer.py --incremental # Skip unchanged files
"""

import hashlib
import os
import uuid
import sys
import json
from datetime import datetime, timezone

from pathspec import PathSpec
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
    BM25_INDEX_PATH,
)
from logger import logger
from symbol_parser import extract_symbols_from_file

import repositories

import pickle
from rank_bm25 import BM25Okapi

INDEX_FILE_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Qdrant client and embedder (module-level singletons)
# ---------------------------------------------------------------------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Shared progress state
# ---------------------------------------------------------------------------
# Written by index_directory(), read by /index/progress SSE endpoint.
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


def collect_indexable_files(
    directory: str,
    max_bytes: int | None = None,
    gitignore_rules: list[tuple[str, PathSpec]] | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Recursively collect files eligible for indexing from a directory.

    Filters by SUPPORTED_EXTENSIONS and MAX_FILE_SIZE_BYTES. Files that are
    unreadable or exceed the size limit are recorded in the skipped list.
    When *gitignore_rules* is provided (see load_gitignore_rules()), files
    and directories matched by any applicable .gitignore rule are excluded
    and recorded as skipped with reason "gitignored".

    Args:
        directory: Root directory to scan.
        max_bytes: Optional override for the max file size limit (bytes).
        gitignore_rules: Optional list of (base_dir, PathSpec) pairs. Paths
            are matched relative to each rule's own base_dir, mirroring git's
            handling of nested .gitignore files.

    Returns:
        Tuple of (files_to_index, skipped_files).
        skipped_files is a list of dicts with keys: path, size_bytes, reason.
    """
    limit = MAX_FILE_SIZE_BYTES if max_bytes is None else max_bytes
    files_to_index, skipped = [], []
    rules = gitignore_rules or []

    for root, dirnames, files in os.walk(directory):
        # Prune ignored directories so we don't descend into them.
        # A trailing separator makes directory-only patterns (e.g. "build/")
        # match the directory itself.
        dirnames[:] = [
            d for d in dirnames
            if not _is_gitignored(os.path.join(root, d) + os.sep, rules)
        ]

        for file in files:
            if file == ".gitignore":
                continue  # .gitignore files are never indexed

            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                path = os.path.join(root, file)
                logger.info(
                    "Skipping unsupported file %s (extension %r not in %s)",
                    path,
                    ext,
                    sorted(SUPPORTED_EXTENSIONS),
                )
                skipped.append({
                    "path": path,
                    "size_bytes": None,
                    "reason": "unsupported_extension",
                    "detail": ext if ext else "(no extension)",
                })
                continue

            path = os.path.join(root, file)
            if _is_gitignored(path, rules):
                logger.info("Skipping gitignored file %s", path)
                skipped.append({
                    "path": path,
                    "size_bytes": None,
                    "reason": "gitignored",
                    "detail": "matched by .gitignore rule",
                })
                continue

            try:
                size = os.path.getsize(path)
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                skipped.append({
                    "path": path,
                    "size_bytes": None,
                    "reason": "unreadable",
                    "detail": str(exc),
                })
                continue

            if size > limit:
                logger.warning(
                    "Skipping oversized file %s (%.2f MB exceeds %.2f MB limit)",
                    path,
                    size / (1024 * 1024),
                    limit / (1024 * 1024),
                )
                skipped.append({
                    "path": path,
                    "size_bytes": size,
                    "reason": "oversized",
                    "detail": (
                        f"{size / (1024 * 1024):.2f} MB exceeds "
                        f"{limit / (1024 * 1024):.2f} MB limit"
                    ),
                })
                continue

            files_to_index.append(path)

    return files_to_index, skipped


def create_collection(collection_name: str) -> None:
    """
    (Re)create the Qdrant collection with cosine distance and the configured embedding dimension.

    Deletes the existing collection if it already exists to ensure a clean state.
    """
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection may not exist; safe to ignore.

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=Distance.COSINE,
        ),
    )


def chunk_file(filepath: str, chunk_size: int = INDEX_CHUNK_SIZE) -> list[dict]:
    """
    Split a source file into overlapping line-based chunks.

    Each chunk includes metadata: original filename and starting line number.
    Empty chunks (after stripping whitespace) are discarded.

    Args:
        filepath: Path to the source file.
        chunk_size: Number of lines per chunk.

    Returns:
        List of chunk dicts with keys: text, file, start_line, is_symbol.

    Raises:
        ValueError: If chunk_size is not greater than INDEX_CHUNK_OVERLAP.
    """
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
    traditional line-based chunks. For other supported extensions only
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


def index_directory(directory: str, repo_id: str | None = None, dry_run: bool = False) -> dict | None:  # noqa: C901
    """
    Main indexing pipeline: scan, chunk, embed, and store codebase into Qdrant + BM25.

    Supports incremental mode via `--incremental` CLI flag, and dry run mode via
    `--dry-run` CLI flag or `dry_run=True` parameter.

    In dry run mode:
      - Validates repository and scans indexable files.
      - Calculates chunk & symbol statistics without creating embeddings or uploading vectors.
      - Returns and logs detailed dry-run summary without modifying vector store or disk caches.
    """
    if "--dry-run" in sys.argv:
        dry_run = True

    # ── Resolve per-repository artifact names ──────────────────────────
    if repo_id is not None:
        target_collection = repositories.collection_name(repo_id)
        target_bm25 = repositories.bm25_path(repo_id)
        target_cache = repositories.cache_path(repo_id)
    else:
        target_collection = QDRANT_COLLECTION_NAME
        target_bm25 = BM25_INDEX_PATH
        target_cache = ".index_cache.json"

    repo_name = os.path.basename(os.path.normpath(directory))

    progress_state.update({
        "running": True,
        "current": 0,
        "total": 0,
        "percent": 0,
        "current_file": "",
        "status": "running",
        "message": "Starting (Dry Run)..." if dry_run else "Starting...",
    })

    # ── Collect .gitignore rules (root + nested) ────────────────────────
    gitignore_rules = load_gitignore_rules(directory)

    try:
        # ── Load previous cache for incremental mode ──────────────────────
        before_cache_data = get_before_cache_data(target_cache)

        if isinstance(before_cache_data, dict):
            metadata = before_cache_data.get("_metadata")
            if metadata and isinstance(metadata, dict):
                repo_version = metadata.get("embedding_version")
                if repo_version and repo_version != EMBEDDING_VERSION:
                    logger.warning(
                        f"Embedding version mismatch detected (repository version: {repo_version}, "
                        f"configured version: {EMBEDDING_VERSION}). Re-indexing is recommended."
                    )

        # ── Prepare Qdrant collection ─────────────────────────────────────
        if not dry_run:
            if "--incremental" not in sys.argv or not client.collection_exists(target_collection):
                create_collection(target_collection)

        points = []
        total_uploaded = 0
        cache_data = {}

        def upsert_pending(indexed_files: int) -> None:
            """Upload and release the current bounded point batch."""
            nonlocal points, total_uploaded
            if not points:
                return

            batch = points
            points = []
            client.upsert(
                collection_name=target_collection,
                points=batch,
            )
            total_uploaded += len(batch)
            print(f"Indexed {indexed_files}/{total_files} files")

        # ── Collect eligible files ────────────────────────────────────────
        all_files, skipped_files = collect_indexable_files(
            directory, gitignore_rules=gitignore_rules
        )
        progress_state["skipped"] = skipped_files
        progress_state["skipped_count"] = len(skipped_files)
        total_files = len(all_files)
        progress_state["total"] = total_files
        skip_msg = f" ({len(skipped_files)} skipped)" if skipped_files else ""
        progress_state["message"] = f"Found {total_files} file(s) to index{skip_msg}"

        # ── Process each file ────────────────────────────────────────────
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
                "hash": get_file_hash(path),
            }

            # Skip unchanged files in incremental mode
            if "--incremental" in sys.argv and path in before_cache_data:
                if is_cache_unchanged(before_cache_data, cache_data, path):
                    cache_data[path]["symbols"] = before_cache_data[path].get("symbols", [])
                    if idx % INDEX_FILE_BATCH_SIZE == 0 or idx == total_files:
                        upsert_pending(idx)
                    continue

            chunks = get_file_chunks(path)
            file_symbols = []
            for chunk in chunks:
                if chunk.get("is_symbol"):
                    file_symbols.append({
                        "name": chunk["symbol_name"],
                        "type": chunk["symbol_type"]
                    })
            cache_data[path]["symbols"] = file_symbols

            line_count = sum(1 for c in chunks if not c.get("is_symbol"))
            sym_count = sum(1 for c in chunks if c.get("is_symbol"))
            msg = f" {file} → {line_count} line chunks"
            if sym_count:
                msg += f", {sym_count} symbols"
            print(msg)

            if not dry_run:
                for chunk in chunks:
                    chunk["repository"] = repo_name
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

                if idx % INDEX_FILE_BATCH_SIZE == 0 or idx == total_files:
                    upsert_pending(idx)

        total_chunks = 0
        total_symbols = 0
        for p, data in cache_data.items():
            total_symbols += len(data.get("symbols", []))

        if dry_run:
            # Aggregate chunk stats across files
            for p in all_files:
                file_chunks = get_file_chunks(p)
                total_chunks += len(file_chunks)

            dry_run_summary = {
                "dry_run": True,
                "repository": repo_name,
                "total_files": total_files,
                "skipped_files_count": len(skipped_files),
                "estimated_chunks": total_chunks,
                "total_symbols": total_symbols,
                "skipped_files": skipped_files,
                "vectors_uploaded": 0,
            }
            logger.info("Indexing Dry Run Summary for %s: %s", repo_name, dry_run_summary)
            print(
                f"\n[DRY RUN COMPLETE] Evaluated {total_files} file(s), "
                f"{total_chunks} chunk(s), {total_symbols} symbol(s), "
                f"{len(skipped_files)} skipped file(s). No vectors were uploaded."
            )
            progress_state.update({
                "running": False,
                "percent": 100,
                "status": "done",
                "message": f"Dry run complete. {total_files} file(s) evaluated ({total_chunks} chunks previewed, 0 vectors uploaded).",
                "dry_run_summary": dry_run_summary,
            })
            return dry_run_summary

        if total_uploaded:
            print(
                f"\nDone. Indexed {total_uploaded} total chunks into Qdrant."
            )
        else:
            print("\nNo changes detected. Nothing to upsert.")

        # ── Build BM25 keyword index ─────────────────────────────────────
        all_chunks = []
        for path in all_files:
            chunks = get_file_chunks(path)
            for chunk in chunks:
                chunk["repository"] = repo_name
            all_chunks.extend(chunks)

        if all_chunks:
            tokenize_corpus = [tokenize(c["text"]) for c in all_chunks]
            bm25 = BM25Okapi(tokenize_corpus)
            with open(target_bm25, "wb") as f:
                pickle.dump({
                    "bm25": bm25,
                    "corpus": [c["text"] for c in all_chunks],
                    "chunks": all_chunks,
                }, f)
            print(f"BM25 index saved ({len(all_chunks)} chunks) to {target_bm25}")

        # ── Dependency Summary ───────────────────────────────────────────
        from dependency_parser import generate_dependency_summary
        dep_summary = generate_dependency_summary(directory)
        logger.info("Detected %d unique dependencies in repository %s", dep_summary["total_unique_dependencies"], repo_name)
        if dep_summary["all_dependencies"]:
            logger.info("Dependencies: %s", ", ".join(dep_summary["all_dependencies"]))

        # ── Save cache metadata ──────────────────────────────────────────
        cache_data["_metadata"] = {
            "repository_name": os.path.basename(os.path.abspath(directory)),
            "indexing_timestamp": datetime.now(timezone.utc).isoformat(),
            "indexed_file_count": len(cache_data),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_version": EMBEDDING_VERSION,
            "skipped_files": skipped_files,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "dependency_summary": dep_summary,
        }

        with open(target_cache, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        skip_summary = f", {len(skipped_files)} file(s) skipped" if skipped_files else ""
        dep_msg = f", {dep_summary['total_unique_dependencies']} dependencies detected" if dep_summary["total_unique_dependencies"] > 0 else ""
        progress_state.update({
            "running": False,
            "percent": 100,
            "status": "done",
            "message": f"Indexing complete. {total_files} file(s) processed{skip_summary}{dep_msg}, {total_uploaded} chunks uploaded.",
            "dependency_summary": dep_summary,
        })

    except Exception as e:
        progress_state.update({
            "running": False,
            "status": "error",
            "message": f"Indexing failed: {e}",
        })
        raise


def get_file_hash(filepath: str) -> str:
    """
    Compute the MD5 hash of a file for change detection.

    Reads the file in 4KB chunks to handle large files efficiently.

    Args:
        filepath: Path to the file.

    Returns:
        Hexadecimal MD5 digest string.
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def tokenize(text: str) -> list[str]:
    """
    Split code text into lowercased word tokens for BM25 indexing.

    Uses regex word-boundary matching to extract alphanumeric tokens.

    Args:
        text: Raw code or natural language text.

    Returns:
        List of lowercased word tokens.
    """
    import re
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]


def load_gitignore_rules(root: str) -> list[tuple[str, PathSpec]]:
    """
    Collect .gitignore rules from *root* and all of its subdirectories.

    Walks the tree once and returns one (base_dir, PathSpec) pair per
    .gitignore file found, with each file's patterns interpreted relative to
    its own directory — mirroring git's handling of nested .gitignore files.
    A file is considered ignored when any applicable spec matches it.

    Args:
        root: Directory to scan for .gitignore files.

    Returns:
        List of (base_dir, PathSpec) tuples. Empty if no .gitignore exists.
    """
    rules = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if ".gitignore" not in filenames:
            continue
        gitignore_path = os.path.join(dirpath, ".gitignore")
        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                spec = PathSpec.from_lines("gitignore", f)
        except OSError as exc:
            logger.warning("Could not read %s: %s", gitignore_path, exc)
            continue
        rules.append((dirpath, spec))
    return rules

def get_before_cache_data(path: str) -> dict:
    """
    Load the previous run's cache data from *path*.

    Returns:
        Dictionary of cached file metadata, or an empty dict if the file
        is missing or unreadable.
    """
    before_cache_data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                before_cache_data = json.load(f)
        except Exception:
            logger.warning("Corrupted .index_cache.json found.")
    return before_cache_data

def is_cache_unchanged(before_cache_data: dict, cache_data: dict, path: str) -> bool:
    """
    Return True if *path*'s file is unchanged since the previous index run.

    Compares the stored mtime first (within a 1ms tolerance); if that differs,
    falls back to comparing the file hash. Used by incremental indexing to
    skip files that have not changed.

    Args:
        before_cache_data: Cache data from the previous run.
        cache_data: Cache data being built for the current run.
        path: File to compare.

    Returns:
        True if the file is unchanged (can be skipped), else False.
    """
    if abs(before_cache_data[path]["mtime"] - cache_data[path]["mtime"]) <= 0.001:
        return True
    else:
        if before_cache_data[path]["hash"] == cache_data[path]["hash"]:
            return True
    return False

def _is_gitignored(path: str, rules: list[tuple[str, PathSpec]]) -> bool:
    """
    Return True if *path* is matched by any applicable .gitignore rule.

    Each rule is a (base_dir, PathSpec) pair; *path* is matched relative to
    that rule's base_dir, so nested .gitignore files only affect files below
    their own directory. Negation patterns (lines starting with ``!``) inside
    a single .gitignore file are honored; cross-file negation is best-effort.
    """
    for base_dir, spec in rules:
        try:
            rel_path = os.path.relpath(path, base_dir).replace("\\", "/")
        except ValueError:
            continue  # different drive (Windows); rule cannot apply
        if rel_path == ".." or rel_path.startswith("../"):
            continue  # path lies outside this .gitignore's directory
        if spec.match_file(rel_path):
            return True
    return False


if __name__ == "__main__":
    index_directory(SAMPLE_CODEBASE_DIRECTORY)
