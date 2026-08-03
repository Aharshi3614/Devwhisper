"""
main.py — FastAPI webhook server and orchestration layer for DevWhisper.

This module is the central entry point for the DevWhisper voice agent.
It exposes REST endpoints that receive webhooks from Vapi (voice platform),
orchestrates the retrieval + LLM pipeline, manages per-session conversation
memory, and provides admin/monitoring utilities.

Endpoints:
    POST /          — Health/root check (prevents 502 from load balancers).
    POST /webhook   — Main Vapi webhook handler (assistant-request, function-call, tool-calls).
    GET  /health    — Liveness probe.
    GET  /statistics— Repository and indexing statistics.
    POST /reset     — Clear all conversation history.
    POST /stream    — Streaming query endpoint (SSE-style text/plain).
    GET  /admin/sessions — List active conversation sessions (admin only).
    POST /index/start    — Trigger background indexing.
    GET  /index/progress — SSE stream of indexing progress.
    GET  /history        — Retrieve conversation history.

Architecture:
    Vapi (voice) → FastAPI (/webhook) → Retriever (hybrid search) →
    LLM (Groq) → FastAPI response → Vapi (TTS)

Security:
    Admin endpoints are protected by an X-Admin-Secret header and fail
    closed (401) if ADMIN_SECRET is not configured.
"""

from fastapi import FastAPI, Request, Header, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from retriever import retrieve, embedder, client as qdrant_client, get_repository_metadata
from llm import generate_response, generate_response_stream
from cache import get as cache_get, put as cache_put
from handlers import route_command
from logger import logger
from errors import error_response
from indexer import index_directory, progress_state, get_before_cache_data, collect_indexable_files, \
    load_gitignore_rules, get_file_hash, is_cache_unchanged
from session_manager import SessionManager
from config import SAMPLE_CODEBASE_DIRECTORY, QDRANT_COLLECTION_NAME
import json
import os
import time
import asyncio
import shutil
import zipfile
import threading
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DevWhisper API",
    description="Voice-native developer experience agent — webhook server and query API.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Admin secret (fail-closed security)
# ---------------------------------------------------------------------------
# Read once at startup. If unset, the /admin/* endpoints fail closed
# (always 401) so sessions are never accidentally exposed.
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "").strip()

# Serve static files (e.g., frontend assets) from the ./static directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Per-session memory store
# ---------------------------------------------------------------------------
# In-memory LRU cache for conversation history.
# Structure: { session_id: {"history": [...], "last_used": <timestamp>} }
# Protected by session_lock for thread safety.
MAX_SESSIONS = 100
MAX_HISTORY_PER_SESSION = 5

session_manager = SessionManager(
    max_sessions=MAX_SESSIONS,
    max_history_per_session=MAX_HISTORY_PER_SESSION,
)

# Backward-compatible aliases used by existing endpoints and tests.
conversation_sessions = session_manager.sessions
session_lock = session_manager.lock


# Detected at startup: whether the codebase changed since the last index run.
# Read by the /index/change endpoint so the frontend can show a re-index hint.
reindex_recommended = False

# TTL cache for /index/change: re-scan at most once every N seconds so the
# endpoint stays cheap to poll while still noticing changes mid-run.
REINDEX_CHECK_INTERVAL = 60  # seconds
_reindex_last_checked_at = 0.0


@app.on_event("startup")
async def startup_event():
    """Warm up the embedder and check whether re-indexing is recommended."""
    global reindex_recommended, _reindex_last_checked_at
    reindex_recommended = is_repository_change()
    _reindex_last_checked_at = time.time()  # start the TTL clock at startup
    if reindex_recommended:
        logger.warning("Codebase changed. Re-indexing is recommended.")
    embedder.encode("warmup query")
    logger.info("Embedder warmed up and ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close the Qdrant client connection on server shutdown."""
    logger.info("Shutting down DevWhisper server...")
    try:
        qdrant_client.close()
        logger.info("Qdrant client connection closed successfully.")
    except Exception:
        logger.error("Error during Qdrant client connection cleanup", exc_info=True)


def _get_session_id(message: dict) -> str:
    """
    Extract a stable session / call ID from the Vapi payload.

    Tries multiple fallback fields because Vapi payloads vary by event type.
    Falls back to "default" if no ID is found.

    Args:
        message: The Vapi message dict (from webhook body).

    Returns:
        A session identifier string.
    """
    call = message.get("call", {})
    if isinstance(call, dict) and call.get("id"):
        return call["id"]
    if message.get("callId"):
        return message["callId"]
    if message.get("sessionId"):
        return message["sessionId"]
    return "default"


def _evict_if_needed():
    """Evict least-recently-used sessions when over capacity."""
    session_manager.evict_if_needed()


def update_memory(session_id: str, user: str, assistant: str) -> None:
    """Append an exchange to a session's bounded history."""
    session_manager.update(session_id, user, assistant)


def get_memory(session_id: str) -> str:
    """Return formatted history for a session."""
    return session_manager.get(session_id)


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@app.post("/")
async def root():
    """
    Root route to prevent 502 errors from health-checking load balancers.

    Returns:
        {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/webhook")
async def vapi_webhook(request: Request):
    """
    Main Vapi webhook handler.

    Handles three Vapi event types:
      1. assistant-request — Returns assistant configuration (first message, model, voice).
      2. function-call    — Legacy single-function call from Vapi.
      3. tool-calls       — Modern multi-tool call from Vapi.

    For function/tool calls, the pipeline is:
      cache lookup → (miss) retrieve(context) → LLM → cache store → response

    Args:
        request: FastAPI Request object containing the Vapi JSON payload.

    Returns:
        JSONResponse with assistant config, tool results, or error.
    """
    try:
        body = await request.json()
        logger.info("Incoming webhook payload: %s", body)

        message = body.get("message", {})
        msg_type = message.get("type", "")
        session_id = _get_session_id(message)

        # ── Assistant initialization ──────────────────────────────────────
        if msg_type == "assistant-request":
            return JSONResponse({
                "assistant": {
                    "firstMessage": "Hey, DevWhisper here. What are you building or debugging?",
                    "model": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "functions": [{
                            "name": "query_codebase",
                            "description": "Search and explain code or debug errors",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }]
                    },
                    "voice": {"provider": "11labs", "voiceId": "burt"}
                }
            })

        # ── Function / tool call handling ─────────────────────────────────
        if msg_type in ["function-call", "tool-calls"]:
            tools = []

            if msg_type == "function-call":
                tools = [{
                    "id": "single",
                    "function": message.get("functionCall", {})
                }]
            else:
                tools = message.get("toolCalls", [])

            results = []

            for tool in tools:
                fn = tool.get("function", {})
                fn_name = fn.get("name", "")

                # Handle both dict and stringified JSON parameters.
                params = fn.get("arguments") or fn.get("parameters") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse command parameters: %s", e)
                        return error_response(400, "Invalid JSON in command parameters. Try rephrasing.")

                if fn_name == "query_codebase":
                    query = params.get("query", "")
                    if not query:
                        return error_response(400, "Query parameter is required and cannot be empty.")

                    # ── Cache lookup ──────────────────────────────────────
                    # Attempt to serve the response from cache. This skips
                    # retrieval and LLM generation entirely on a hit.
                    cached = cache_get(query)
                    if cached is not None:
                        # Still update conversation memory on cache hit so
                        # the session history stays consistent.
                        update_memory(session_id, query, cached)
                        results.append({
                            "toolCallId": tool.get("id", "single"),
                            "result": cached
                        })
                        continue

                    # ── Cache miss: run full pipeline ─────────────────────
                    context, sources = retrieve(query, include_sources=True)
                    history = get_memory(session_id)
                    answer = route_command(query, session_id) or generate_response(query, context, history)

                    if answer and answer.strip() and sources:
                        answer += "\n\n**Sources used:** " + ", ".join(f"`{s}`" for s in sources)

                    # ── Cache insertion ─────────────────────────────────
                    # Only cache successful, non-empty responses.
                    if answer and answer.strip():
                        cache_put(query, answer)

                    update_memory(session_id, query, answer)

                    results.append({
                        "toolCallId": tool.get("id", "single"),
                        "result": answer
                    })

            return JSONResponse({"results": results})

        return JSONResponse({"status": "ok"})

    except Exception:
        logger.error("SERVER ERROR", exc_info=True)
        return error_response(500, "An unexpected server error occurred. Please try again.")


@app.get("/health")
def health():
    """
    Liveness / readiness probe.

    Returns:
        {"status": "ok", "message": "DevWhisper is running"}
    """
    return {"status": "ok", "message": "DevWhisper is running"}


@app.get("/statistics")
def get_statistics():
    """
    Return repository and indexing statistics.

    Queries Qdrant for collection metadata and reads the local
    .index_cache.json for indexed file counts.

    Returns:
        JSON with indexed_file_count, chunk_count, and collection_info.
    """
    try:
        try:
            collection_info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
            collection_dict = (
                collection_info.model_dump() if hasattr(collection_info, "model_dump")
                else collection_info.dict() if hasattr(collection_info, "dict")
                else vars(collection_info)
            )
            chunk_count = getattr(collection_info, "points_count", None)
            if chunk_count is None:
                chunk_count = collection_dict.get("points_count", 0)
        except Exception as e:
            logger.warning("Failed to get collection info from Qdrant: %s", e)
            collection_dict = {}
            chunk_count = 0

        metadata = get_repository_metadata()
        indexed_file_count = metadata.get("indexed_file_count", 0)

        return {
            "indexed_file_count": indexed_file_count,
            "chunk_count": chunk_count,
            "collection_info": collection_dict,
        }
    except Exception:
        logger.error("Failed to retrieve statistics", exc_info=True)
        return error_response(500, "Failed to retrieve statistics.")


@app.post("/reset")
def reset_memory():
    """
    Clear all in-memory conversation history.

    Returns:
        {"status": "memory cleared"}
    """
    session_manager.clear()
    return {"status": "memory cleared"}


@app.post("/stream")
async def stream_query(request: Request):
    """
    Streaming query endpoint.

    Accepts a JSON body with { "query": "...", "sessionId": "..." } and
    returns a text/plain StreamingResponse with tokens yielded as they are
    generated by the LLM. Falls back to cache if available.

    Args:
        request: FastAPI Request with JSON body.

    Returns:
        StreamingResponse (text/plain) with the generated answer.
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        session_id = body.get("sessionId", "default")

        if not query:
            return error_response(400, "Query parameter is required and cannot be empty.")

        # Cache lookup
        cached = cache_get(query)
        if cached is not None:
            update_memory(session_id, query, cached)

            async def cached_generator():
                yield cached

            return StreamingResponse(cached_generator(), media_type="text/plain")

        # Cache miss: run retrieval
        context, sources = retrieve(query, include_sources=True)
        history = get_memory(session_id)

        def event_generator():
            full_response = []
            for token in generate_response_stream(query, context, history):
                full_response.append(token)
                yield token

            if sources:
                sources_str = "\n\n**Sources used:** " + ", ".join(f"`{s}`" for s in sources)
                yield sources_str
                full_response.append(sources_str)

            # Update cache and session history on complete stream
            answer = "".join(full_response)
            if answer and answer.strip():
                cache_put(query, answer)
                update_memory(session_id, query, answer)

        return StreamingResponse(event_generator(), media_type="text/plain")

    except Exception:
        logger.error("SERVER STREAM ERROR", exc_info=True)
        return error_response(500, "An unexpected server error occurred in the stream. Please try again.")


# ---------------------------------------------------------------------------
# Admin endpoints (protected by X-Admin-Secret)
# ---------------------------------------------------------------------------

def _require_admin(x_admin_secret: str | None) -> None:
    """
    Validate the X-Admin-Secret header against the server's ADMIN_SECRET.

    Uses a direct equality comparison. Raises 401 on any mismatch (including
    missing header or unset server secret) so the endpoint never leaks session
    data by accident.

    Args:
        x_admin_secret: Value of the X-Admin-Secret HTTP header.

    Raises:
        HTTPException: 401 if secret is missing, not configured, or invalid.
    """
    if not ADMIN_SECRET:
        # Fail closed if the operator forgot to set ADMIN_SECRET.
        raise HTTPException(status_code=401, detail="Admin secret not configured on server")
    if not x_admin_secret or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing admin secret")


@app.get("/admin/sessions")
def admin_list_sessions(x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret")):
    """
    Return the current set of active conversation sessions.

    Reads from the in-memory `conversation_sessions` store used by the
    webhook pipeline. Intended for debugging / monitoring only — no PII
    beyond session IDs and timestamps is exposed.

    Headers:
        X-Admin-Secret: Must match the server's ADMIN_SECRET env var.

    Returns:
        JSON with session list, active count, max capacity, and generation timestamp.
    """
    _require_admin(x_admin_secret)

    now = time.time()
    sessions = []
    with session_lock:
        for session_id, data in conversation_sessions.items():
            last_used_ts = data.get("last_used", 0)
            try:
                last_used_iso = (
                    datetime.fromtimestamp(last_used_ts, tz=timezone.utc).isoformat()
                    .replace("+00:00", "Z")
                )
            except (ValueError, OSError):
                # Defensive: skip malformed timestamps rather than fail the request.
                last_used_iso = None

            sessions.append(
                {
                    "session_id": session_id,
                    "last_used": last_used_iso,
                    "last_used_ago_seconds": int(now - last_used_ts) if last_used_ts else None,
                    "message_count": len(data.get("history", [])),
                }
            )

    # Sort by most-recently-used first — most useful for monitoring.
    sessions.sort(key=lambda s: s["last_used_ago_seconds"] or float("inf"))

    return {
        "status": "ok",
        "active_sessions": len(sessions),
        "max_sessions": MAX_SESSIONS,
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "sessions": sessions,
    }


@app.post("/index/start")
def start_indexing():
    """
    Trigger codebase indexing in a background thread.

    Returns:
        {"status": "started", "message": "..."} or 409 if already running.
    """
    if progress_state.get("running"):
        return error_response(409, "Indexing is already in progress.")
    threading.Thread(target=index_directory, args=(SAMPLE_CODEBASE_DIRECTORY,), daemon=True).start()
    return {"status": "started", "message": "Indexing started. Poll /index/progress for updates."}


@app.post("/index/upload")
async def upload_codebase(file: UploadFile = File(...)):
    """
    Accept a ZIP archive of a local codebase, extract it, and automatically index it.
    """
    if progress_state.get("running"):
        return error_response(409, "Indexing is already in progress.")

    if not file.filename.endswith(".zip"):
        return error_response(400, "Only ZIP archives are supported.")

    # Temporarily save the zip file on disk
    temp_zip_path = "temp_upload.zip"
    try:
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to write uploaded file to disk: %s", e)
        return error_response(500, f"Failed to save uploaded file: {e}")

    # Validate ZIP file structure
    if not zipfile.is_zipfile(temp_zip_path):
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        return error_response(400, "Invalid ZIP archive.")

    is_path_traversal = False
    invalid_member = ""
    try:
        # Check Zip Slip path traversal
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                # Avoid relative paths pointing outside target
                norm_path = os.path.normpath(member)
                if norm_path.startswith("..") or os.path.isabs(norm_path):
                    is_path_traversal = True
                    invalid_member = member
                    break

            if not is_path_traversal:
                # Recreate target directory cleanly to ensure isolation
                if os.path.exists(SAMPLE_CODEBASE_DIRECTORY):
                    shutil.rmtree(SAMPLE_CODEBASE_DIRECTORY)
                os.makedirs(SAMPLE_CODEBASE_DIRECTORY, exist_ok=True)

                # Extract the ZIP
                zip_ref.extractall(SAMPLE_CODEBASE_DIRECTORY)
    except Exception as e:
        logger.error("Error during ZIP validation/extraction: %s", e)
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except Exception:
                pass
        return error_response(500, f"Failed to process ZIP archive: {e}")

    # Clean up temp file
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)

    if is_path_traversal:
        return error_response(400, f"Path traversal detected in ZIP: {invalid_member}")

    # Validate that extracted directory has supported files (.py, .md)
    has_supported_file = False
    for root, dirs, files in os.walk(SAMPLE_CODEBASE_DIRECTORY):
        for f in files:
            if f.endswith((".py", ".md")):
                has_supported_file = True
                break
        if has_supported_file:
            break

    if not has_supported_file:
        # Clean up the directory since it's invalid/unsupported
        if os.path.exists(SAMPLE_CODEBASE_DIRECTORY):
            shutil.rmtree(SAMPLE_CODEBASE_DIRECTORY)
        return error_response(400, "No supported files (.py, .md) found in the uploaded ZIP archive.")

    # Trigger background indexing
    threading.Thread(target=index_directory, args=(SAMPLE_CODEBASE_DIRECTORY,), daemon=True).start()

    return {
        "status": "started",
        "message": "ZIP file uploaded and extracted successfully. Indexing started in the background."
    }


@app.get("/index/progress")
async def index_progress():
    """
    Server-Sent Events (SSE) stream of indexing progress.

    Emits the current progress_state as JSON every 0.5s until indexing
    completes (status: done, error, or idle).

    Returns:
        StreamingResponse with media_type="text/event-stream".
    """
    async def event_stream():
        while True:
            state = dict(progress_state)
            yield f"data: {json.dumps(state)}\n\n"
            if state["status"] in ("done", "error", "idle"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history")
def get_history(session_id: str | None = None):
    """
    Retrieve conversation history.

    - GET /history              → returns all active session IDs.
    - GET /history?session_id=xxx → returns history for that specific session.

    Args:
        session_id: Optional session identifier to filter by.

    Returns:
        JSON with either session_ids list or session_id + history array.
    """
    with session_lock:
        if session_id:
            session = conversation_sessions.get(session_id)
            history = list(session["history"]) if session else []
            return {"session_id": session_id, "history": history}

        sessions_info = []
        for sid, data in conversation_sessions.items():
            history_list = data.get("history", [])
            preview = ""
            if history_list:
                first_entry = history_list[0]
                user_line = next(
                    (line for line in first_entry.split("\n") if line.startswith("User: ")),
                    None,
                )
                preview = user_line[6:] if user_line else ""
            sessions_info.append({
                "session_id": sid,
                "last_used": data.get("last_used", 0),
                "message_count": len(history_list),
                "preview": preview,
            })

        all_session_ids = list(conversation_sessions.keys())
    return {"session_ids": all_session_ids, "sessions": sessions_info}

def is_repository_change() -> bool:
    """
    Return True if the codebase has changed since the last indexing run.

    Collects the current indexable files (respecting .gitignore) and compares
    each file's mtime and hash against the cached values in .index_cache.json.
    Returns True as soon as any file differs — meaning re-indexing is
    recommended.

    Returns:
        True if the repository changed since the last index, else False.
    """
    cache_data = {}
    gitignore_rules = load_gitignore_rules(SAMPLE_CODEBASE_DIRECTORY)
    all_files, _skipped = collect_indexable_files(
        SAMPLE_CODEBASE_DIRECTORY, gitignore_rules=gitignore_rules
    )
    before_cache_data = get_before_cache_data()
    for path in all_files:
        cache_data[path] = {
            "mtime": os.path.getmtime(path),
            "hash": get_file_hash(path),
        }
        if path not in before_cache_data:
            return True  # new file, not in cache → changed
        if not is_cache_unchanged(before_cache_data, cache_data, path):
            return True
    return False


@app.get("/index/change")
def index_change_recommendation():
    """
    Return whether re-indexing is recommended.

    Re-scanning the codebase is expensive (walking every file + hashing), so
    this endpoint caches the result and only recomputes it once every
    REINDEX_CHECK_INTERVAL seconds. Cheap to poll, still notices changes.

    Returns:
        Dict with `reindex_recommended` (bool) and `message` (str).
    """
    global reindex_recommended, _reindex_last_checked_at
    now = time.time()
    if now - _reindex_last_checked_at >= REINDEX_CHECK_INTERVAL:
        reindex_recommended = is_repository_change()
        _reindex_last_checked_at = now
    return {
        "reindex_recommended": reindex_recommended,
        "message": "Codebase changed. Re-indexing is recommended." if reindex_recommended
        else "Index is up to date.",
    }


@app.get("/index/suggestions")
def get_index_suggestions():
    """
    Generate contextual query suggestions based on the indexed codebase.
    """
    suggestions = []

    cache_path = ".index_cache.json"
    cache_data = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            pass

    files_info = []
    for filepath, info in cache_data.items():
        if filepath.startswith("_"):
            continue
        filename = os.path.basename(filepath)
        symbols = info.get("symbols", []) if isinstance(info, dict) else []
        files_info.append({
            "filename": filename,
            "symbols": symbols
        })

    import random

    # Generate suggestions from markdown files
    md_files = [f["filename"] for f in files_info if f["filename"].lower().endswith(".md")]
    for f in md_files:
        suggestions.append(f"What is the purpose of {f}?")
        suggestions.append(f"What contents or features are described in {f}?")

    # Generate suggestions from python files and their symbols
    py_files = [f for f in files_info if f["filename"].lower().endswith(".py")]
    for f in py_files:
        filename = f["filename"]
        symbols = f["symbols"]

        suggestions.append(f"What is the main role of {filename} in this codebase?")
        suggestions.append(f"In {filename}, what functions are found?")

        for sym in symbols:
            name = sym.get("name")
            sym_type = sym.get("type", "function")
            if sym_type == "class":
                suggestions.append(f"How is the class {name} implemented in {filename}?")
                suggestions.append(f"What is the purpose of the class {name}?")
            else:
                suggestions.append(f"In {filename}, what does the function {name} do?")
                suggestions.append(f"Explain the purpose of the function {name}.")

    fallback_suggestions = [
        "What is this project about?",
        "What are the main entry points of this codebase?",
        "Explain the overall architecture of the project.",
        "What libraries or dependencies does this project use?",
        "How do I set up and run this codebase?"
    ]

    if len(suggestions) < 4:
        suggestions.extend(fallback_suggestions)

    # De-duplicate
    unique_suggestions = list(dict.fromkeys(suggestions))

    # Shuffle and pick top 4
    random.shuffle(unique_suggestions)
    selected_suggestions = unique_suggestions[:4]

    return {"suggestions": selected_suggestions}

