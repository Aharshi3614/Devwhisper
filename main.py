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
    GET  /index/summary  — Latest repository scan summary (issue #224).
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
from starlette.middleware.base import BaseHTTPMiddleware
from retriever import retrieve, embedder, client as qdrant_client, get_repository_metadata
from llm import generate_response, generate_response_stream
from cache import get as cache_get, put as cache_put
from handlers import route_command
from errors import error_response
from pipeline_validator import PipelineTracker, PipelineStageError
from indexer import index_directory, progress_state, get_before_cache_data, collect_indexable_files, \
    load_gitignore_rules, get_file_hash, is_cache_unchanged
from session_manager import SessionManager
from config import SAMPLE_CODEBASE_DIRECTORY, QDRANT_COLLECTION_NAME
import repositories
import contextvars
import logging
import queue
import uuid
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
# Issue #225: Request Correlation IDs
# ---------------------------------------------------------------------------
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate a unique request ID for every incoming request (issue #225).

    - Checks for an ``X-Request-ID`` header (set by upstream proxies).
    - Otherwise generates a new UUID4.
    - Stores it in the ``request_id_var`` contextvar so all log calls
      within the request scope automatically include it.
    - Adds it to the response headers as ``X-Request-ID``.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


class RequestIDLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically injects the request_id from the
    contextvar into every log call's extra fields (issue #225)."""

    def process(self, msg, kwargs):
        rid = request_id_var.get(None)
        if rid is not None:
            extra = kwargs.get("extra", {})
            if "request_id" not in extra:
                extra["request_id"] = rid
            kwargs["extra"] = extra
        return msg, kwargs


logger = RequestIDLoggerAdapter(logging.getLogger("devwhisper"), {})

# ---------------------------------------------------------------------------
# Admin secret (fail-closed security)
# ---------------------------------------------------------------------------
# Read once at startup. If unset, the /admin/* endpoints fail closed
# (always 401) so sessions are never accidentally exposed.
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "").strip()

# Serve static files (e.g., frontend assets) from the ./static directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

from di import get_session_manager, get_indexing_queue, get_jobs_history

# ---------------------------------------------------------------------------
# Per-session memory store
# ---------------------------------------------------------------------------
# In-memory LRU cache for conversation history.
# Structure: { session_id: {"history": [...], "last_used": <timestamp>} }
# Protected by session_lock for thread safety.
MAX_SESSIONS = 100
MAX_HISTORY_PER_SESSION = 5

session_manager = get_session_manager(max_sessions=MAX_SESSIONS, max_history_per_session=MAX_HISTORY_PER_SESSION)

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


# ---------------------------------------------------------------------------
# Indexing Queue & Worker
# ---------------------------------------------------------------------------
indexing_queue = get_indexing_queue()
jobs_history = get_jobs_history()

def queue_worker():
    global progress_state
    while True:
        try:
            job = indexing_queue.get()
            if job is None:
                break

            job_id = job["id"]
            job["status"] = "running"
            job["started_at"] = time.time()
            job["message"] = f"Indexing {job['name']}..."

            progress_state.update({
                "running": True,
                "current": 0,
                "total": 0,
                "percent": 0,
                "current_file": "",
                "status": "running",
                "message": f"Starting job: {job['name']}",
                "skipped": [],
                "skipped_count": 0,
            })

            try:
                if job["type"] == "upload":
                    temp_zip_path = job["temp_zip_path"]

                    if os.path.exists(SAMPLE_CODEBASE_DIRECTORY):
                        shutil.rmtree(SAMPLE_CODEBASE_DIRECTORY)
                    os.makedirs(SAMPLE_CODEBASE_DIRECTORY, exist_ok=True)

                    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                        zip_ref.extractall(SAMPLE_CODEBASE_DIRECTORY)

                    # Clean up temp file
                    if os.path.exists(temp_zip_path):
                        try:
                            os.remove(temp_zip_path)
                        except Exception:
                            pass
                    directory = SAMPLE_CODEBASE_DIRECTORY
                else:
                    # reindex: scan the registered path for this repository
                    directory = repositories.get_repo_path(job.get("repo_id")) or SAMPLE_CODEBASE_DIRECTORY

                # Run the actual indexing pipeline
                index_directory(directory, repo_id=job.get("repo_id"), dry_run=job.get("dry_run", False))

                job["status"] = "completed"
                job["percent"] = 100
                job["message"] = "Indexing completed successfully."

                progress_state.update({
                    "running": False,
                    "status": "done",
                    "percent": 100,
                    "message": "Indexing completed successfully.",
                })
            except Exception as e:
                logger.error("Job %s failed: %s", job_id, e)
                job["status"] = "failed"
                job["error"] = str(e)
                job["message"] = f"Failed: {e}"

                progress_state.update({
                    "running": False,
                    "status": "error",
                    "message": f"Indexing failed: {e}",
                })
            finally:
                job["finished_at"] = time.time()
                indexing_queue.task_done()
        except Exception as main_err:
            logger.error("Error in queue worker main loop: %s", main_err)
            time.sleep(1)


@app.on_event("startup")
async def startup_event():
    """Warm up the embedder and check whether re-indexing is recommended."""
    threading.Thread(target=queue_worker, daemon=True).start()
    os.makedirs(repositories.OUTPUT_DIRECTORY, exist_ok=True)
    # sample_codebase is the default repo, but it is git-ignored and may be
    # absent on a fresh checkout — don't crash startup when it is missing.
    if not repositories.load_repositories() and os.path.isdir(SAMPLE_CODEBASE_DIRECTORY):
        repositories.add_repository(SAMPLE_CODEBASE_DIRECTORY)
    if repositories.get_current_repo_id() is None:
        first = repositories.load_repositories()
        if first:
            repositories.set_current_repo(first[0]["id"])
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

        # Cache lookup
        # Issue #227: validate pipeline stage order
        tracker = PipelineTracker()
        try:
            tracker.enter("cache_lookup")
        except PipelineStageError as e:
            logger.error("Pipeline stage violation: %s", e)

        cached = cache_get(query)
        if cached is not None:
            update_memory(session_id, query, cached)
            try:
                tracker.enter("cache_insertion")
            except PipelineStageError as e:
                logger.error("Pipeline stage violation: %s", e)

            async def cached_generator():
                yield cached

            return StreamingResponse(cached_generator(), media_type="text/plain")

        # Cache miss: run retrieval
        try:
            tracker.enter("retrieval")
        except PipelineStageError as e:
            logger.error("Pipeline stage violation: %s", e)

        context, sources = retrieve(query, include_sources=True, repo_id=repositories.get_current_repo_id(), repositories=repositories.get_current_repo_name())
        history = get_memory(session_id)

        try:
            tracker.enter("generation")
        except PipelineStageError as e:
            logger.error("Pipeline stage violation: %s", e)

        def event_generator():
            full_response = []
            for token in generate_response_stream(query, context, history):
                full_response.append(token)
                yield token

            try:
                tracker.enter("post_processing")
            except PipelineStageError as e:
                logger.error("Pipeline stage violation: %s", e)

            if sources:
                sources_str = "\n\n**Sources used:** " + ", ".join(f"`{s}`" for s in sources)
                yield sources_str
                full_response.append(sources_str)

            # Update cache and session history on complete stream
            try:
                tracker.enter("cache_insertion")
            except PipelineStageError as e:
                logger.error("Pipeline stage violation: %s", e)

            answer = "".join(full_response)
            if answer and answer.strip():
                cache_put(query, answer)
                update_memory(session_id, query, answer)

        return StreamingResponse(event_generator(), media_type="text/plain")

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


@app.post("/prompt/preview")
async def preview_prompt_endpoint(request: Request):
    """
    Generate prompt preview displaying retrieved context and constructed prompt (Issue #216).
    Disabled by default unless enabled or explicit preview flag is set.
    """
    body = await request.json()
    preview_enabled = body.get("preview_mode")
    if preview_enabled is None:
        from config import PROMPT_PREVIEW_MODE
        preview_enabled = PROMPT_PREVIEW_MODE

    if not preview_enabled:
        raise HTTPException(status_code=403, detail="Prompt preview mode is disabled.")

    user_query = body.get("query", "")
    repo_id = body.get("repo_id")
    context = body.get("context")
    history = body.get("history", "")

    if not context and user_query:
        context = retrieve(query=user_query, repo_id=repo_id)

    from prompt_builder import generate_prompt_preview
    preview = generate_prompt_preview(user_query=user_query, context=context or "", history=history)
    return JSONResponse(status_code=200, content=preview)


@app.get("/statistics")
def get_statistics():
    """
    Return repository and indexing statistics.

    Queries Qdrant for collection metadata and reads the local
    .index_cache.json for indexed file counts.

    Returns:
        JSON with indexed_file_count, chunk_count, and collection_info.
    """
    current_repo_id = repositories.get_current_repo_id()
    if current_repo_id is not None:
        target_collection = repositories.collection_name(current_repo_id)
        target_cache = repositories.cache_path(current_repo_id)
    else:
        target_collection = QDRANT_COLLECTION_NAME
        target_cache = ".index_cache.json"

    try:
        try:
            collection_info = qdrant_client.get_collection(target_collection)
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

        metadata = get_repository_metadata(target_cache)
        indexed_file_count = metadata.get("indexed_file_count", 0)
        dependency_summary = metadata.get("dependency_summary", {})

        return {
            "indexed_file_count": indexed_file_count,
            "chunk_count": chunk_count,
            "collection_info": collection_dict,
            "dependency_summary": dependency_summary,
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
        context, sources = retrieve(query, include_sources=True, repo_id=repositories.get_current_repo_id(), repositories=repositories.get_current_repo_name())
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
def start_indexing(dry_run: bool = False):
    """
    Queue codebase indexing (supports dry_run=True to preview stats without uploading vectors).
    """
    if progress_state.get("running"):
        return error_response(409, "Indexing is already in progress.")
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": "dry_run" if dry_run else "reindex",
        "name": "Manual Dry Run" if dry_run else "Manual Re-index",
        "repo_id": repositories.get_current_repo_id(),
        "dry_run": dry_run,
        "status": "pending",
        "percent": 0,
        "message": "Pending in queue...",
        "created_at": time.time(),
        "started_at": 0.0,
        "finished_at": 0.0,
        "error": None
    }
    jobs_history.append(job)
    indexing_queue.put(job)
    return {"status": "started", "message": "Manual re-indexing job queued.", "job_id": job_id, "dry_run": dry_run}


@app.post("/index/upload")
async def upload_codebase(file: UploadFile = File(...)):
    """
    Accept a ZIP archive of a local codebase, queue it, extract it, and automatically index it sequentially.
    """
    if not file.filename.endswith(".zip"):
        return error_response(400, "Only ZIP archives are supported.")

    job_id = str(uuid.uuid4())
    temp_zip_path = f"temp_upload_{job_id}.zip"
    try:
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to write uploaded file to disk: %s", e)
        return error_response(500, f"Failed to save uploaded file: {e}")

    # Validate ZIP file structure immediately to give quick feedback
    if not zipfile.is_zipfile(temp_zip_path):
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        return error_response(400, "Invalid ZIP archive.")

    # Validate path traversal (Zip Slip) synchronously
    is_path_traversal = False
    invalid_member = ""
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            norm_path = os.path.normpath(member)
            if norm_path.startswith("..") or os.path.isabs(norm_path):
                is_path_traversal = True
                invalid_member = member
                break

    if is_path_traversal:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        return error_response(400, f"Path traversal detected in ZIP: {invalid_member}")

    # Validate that ZIP contains supported files (.py, .md) synchronously
    has_supported_file = False
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if not member.endswith("/") and member.lower().endswith((".py", ".md")):
                has_supported_file = True
                break

    if not has_supported_file:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        return error_response(400, "No supported files (.py, .md) found in the uploaded ZIP archive.")

    # Queue the job
    job = {
        "id": job_id,
        "type": "upload",
        "name": f"Upload: {file.filename}",
        "repo_id": repositories.get_current_repo_id(),
        "temp_zip_path": temp_zip_path,
        "status": "pending",
        "percent": 0,
        "message": "Pending in queue...",
        "created_at": time.time(),
        "started_at": 0.0,
        "finished_at": 0.0,
        "error": None
    }
    jobs_history.append(job)
    indexing_queue.put(job)

    return {
        "status": "started",
        "message": "ZIP file uploaded successfully. Indexing job queued.",
        "job_id": job_id
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


@app.get("/index/queue")
def get_indexing_queue():
    """
    Retrieve the current indexing jobs queue and history.
    """
    return {"jobs": jobs_history}


# ---------------------------------------------------------------------------
# Issue #224: Repository Scan Summary API
# ---------------------------------------------------------------------------
@app.get("/index/summary")
def get_index_summary():
    """
    Return the latest repository scan summary for frontend integration.

    Combines the live in-memory ``progress_state`` (for runs that are
    currently active or just finished) with the persisted ``_metadata``
    block from ``.index_cache.json`` (for the most recent completed run).

    Response shape (always present):
        {
          "repository_id": str | null,
          "repository_name": str | null,
          "status": "idle" | "running" | "done" | "error",
          "indexed_file_count": int,
          "skipped_file_count": int,
          "indexing_duration_seconds": float | null,
          "indexing_timestamp": str | null,     # ISO-8601 UTC, last completed run
          "current_file": str,                   # file being indexed right now (if running)
          "percent": int,                        # 0–100 progress for the active run
          "message": str,
          "skipped_files": list[dict]            # detailed skip reasons (path, reason, detail)
        }

    The endpoint never raises — if the cache file is missing or corrupted
    (e.g. before the first index run), it returns zero counts and null
    duration/timestamp so the frontend can render an "empty state".
    """
    current_repo_id = repositories.get_current_repo_id()
    if current_repo_id is not None:
        target_cache = repositories.cache_path(current_repo_id)
    else:
        target_cache = ".index_cache.json"

    # Defaults — returned when no indexing has happened yet.
    summary = {
        "repository_id": current_repo_id,
        "repository_name": repositories.get_current_repo_name(),
        "status": progress_state.get("status", "idle"),
        "indexed_file_count": 0,
        "skipped_file_count": 0,
        "indexing_duration_seconds": None,
        "indexing_timestamp": None,
        "current_file": progress_state.get("current_file", ""),
        "percent": progress_state.get("percent", 0),
        "message": progress_state.get("message", ""),
        "skipped_files": [],
    }

    # ── Merge persisted metadata from the latest completed run ──────────
    # This is what makes the endpoint useful *after* indexing finishes and
    # progress_state has been reset/clobbered by a later job.
    persisted = get_repository_metadata(target_cache)
    if isinstance(persisted, dict):
        summary["indexed_file_count"] = persisted.get(
            "indexed_file_count", summary["indexed_file_count"]
        )
        summary["skipped_file_count"] = persisted.get(
            "skipped_file_count",
            len(persisted.get("skipped_files", [])) or summary["skipped_file_count"],
        )
        summary["indexing_duration_seconds"] = persisted.get(
            "indexing_duration_seconds", summary["indexing_duration_seconds"]
        )
        summary["indexing_timestamp"] = persisted.get(
            "indexing_timestamp", summary["indexing_timestamp"]
        )
        if persisted.get("repository_name"):
            summary["repository_name"] = persisted["repository_name"]
        summary["skipped_files"] = persisted.get(
            "skipped_files", summary["skipped_files"]
        )

    # ── Overlay live progress for an in-progress / just-finished run ────
    # While indexing is running, progress_state has the freshest counts.
    # After it finishes, the values written above (from _metadata) win
    # because they are the authoritative final numbers.
    if progress_state.get("running"):
        summary["status"] = "running"
        summary["indexed_file_count"] = progress_state.get("current", summary["indexed_file_count"])
        summary["skipped_file_count"] = progress_state.get(
            "skipped_count", summary["skipped_file_count"]
        )
        summary["skipped_files"] = progress_state.get(
            "skipped", summary["skipped_files"]
        )

    return summary

@app.get("/history")
def get_history(session_id: str | None = None):
    """
    Retrieve conversation history.

    - GET /history            → returns all active session IDs.
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
    current_repo_id = repositories.get_current_repo_id()

    directory = SAMPLE_CODEBASE_DIRECTORY
    if current_repo_id is not None:
        found = repositories.get_repo_path(current_repo_id)
        if found:
            directory = found

    cache_data = {}
    cache_file = repositories.cache_path(current_repo_id) if current_repo_id else ".index_cache.json"
    gitignore_rules = load_gitignore_rules(directory)
    all_files, _skipped = collect_indexable_files(
        directory, gitignore_rules=gitignore_rules
    )
    before_cache_data = get_before_cache_data(cache_file)
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

    current_repo_id = repositories.get_current_repo_id()
    cache_path = repositories.cache_path(current_repo_id) if current_repo_id else ".index_cache.json"
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
  

def _is_indexed(repo_id: str) -> bool:
    """Return True if the repository already has an index cache file."""
    return os.path.exists(repositories.cache_path(repo_id))


def _queue_index(repo_id: str) -> None:
    """Queue an auto-indexing job for *repo_id*."""
    if progress_state.get("running"):
        return
    job = {
        "id": str(uuid.uuid4()),
        "type": "reindex",
        "name": f"Auto-index: {repo_id}",
        "repo_id": repo_id,
        "status": "pending",
        "percent": 0,
        "message": "Pending in queue...",
        "created_at": time.time(),
        "started_at": 0.0,
        "finished_at": 0.0,
        "error": None,
    }

    jobs_history.append(job)
    indexing_queue.put(job)


@app.get("/repos")
def list_repositories():
    """List all registered repositories and the currently active one."""
    repos = repositories.load_repositories()
    for repo in repos:
        repo["indexed"] = _is_indexed(repo["id"])
    return {"repos": repos, "current": repositories.get_current_repo_id()}


@app.post("/repos/add")
def add_repository_endpoint(payload: dict):
    """Register a new repository by its server path."""
    path = payload.get("path")
    if not path:
        return error_response(400, "path is required")
    if not os.path.isdir(path):
        return error_response(400, f"Directory not found: {path}")

    repo_id = repositories.add_repository(path)
    if not _is_indexed(repo_id):
        _queue_index(repo_id)
    return {"id": repo_id, "indexed": _is_indexed(repo_id)}


@app.post("/repos/switch")
def switch_repository_endpoint(payload: dict):
    """Switch the active repository, auto-indexing if needed"""
    repo_id = payload.get("repo_id")
    if not repo_id:
        return error_response(400, "repo_id is required")
    repositories.set_current_repo(repo_id)

    if not _is_indexed(repo_id):
        _queue_index(repo_id)
    return {"id": repo_id, "indexed": _is_indexed(repo_id)}

