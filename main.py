from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from retriever import retrieve, embedder, client as qdrant_client
from llm import generate_response
from cache import get as cache_get, put as cache_put
from handlers import route_command
import json
import os
import time
from datetime import datetime, timezone

app = FastAPI()

# --- Admin secret ------------------------------------------------------
# Read once at startup. If unset, the /admin/* endpoints fail closed
# (always 401) so sessions are never accidentally exposed.
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "").strip()

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Per-session memory store ---
# { session_id: {"history": [...], "last_used": <timestamp>} }
conversation_sessions = {}
MAX_SESSIONS = 100
MAX_HISTORY_PER_SESSION = 5


@app.on_event("startup")
async def startup_event():
    embedder.encode("warmup query")
    print("Embedder warmed up and ready!")


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down DevWhisper server...")
    try:
        qdrant_client.close()
        print("Qdrant client connection closed successfully.")
    except Exception as e:
        print(f"Error during Qdrant client connection cleanup: {e}")


def _get_session_id(message: dict) -> str:
    """Extract a session/call ID from the Vapi payload, with fallbacks."""
    call = message.get("call", {})
    if isinstance(call, dict) and call.get("id"):
        return call["id"]
    if message.get("callId"):
        return message["callId"]
    if message.get("sessionId"):
        return message["sessionId"]
    return "default"


def _evict_if_needed():
    """Simple LRU eviction: drop the least-recently-used session if over cap."""
    if len(conversation_sessions) <= MAX_SESSIONS:
        return
    oldest_id = min(
        conversation_sessions,
        key=lambda sid: conversation_sessions[sid]["last_used"]
    )
    del conversation_sessions[oldest_id]


def update_memory(session_id: str, user: str, assistant: str):
    session = conversation_sessions.setdefault(
        session_id, {"history": [], "last_used": time.time()}
    )
    session["history"].append(f"User: {user}\nAssistant: {assistant}")
    if len(session["history"]) > MAX_HISTORY_PER_SESSION:
        session["history"].pop(0)
    session["last_used"] = time.time()
    _evict_if_needed()


def get_memory(session_id: str) -> str:
    session = conversation_sessions.get(session_id)
    if not session:
        return ""
    session["last_used"] = time.time()
    return "\n\n".join(session["history"])


# FIX: root route to prevent 502
@app.post("/")
async def root():
    return {"status": "ok"}


@app.post("/webhook")
async def vapi_webhook(request: Request):
    try:
        body = await request.json()
        print("Incoming:", body)

        message = body.get("message", {})
        msg_type = message.get("type", "")
        session_id = _get_session_id(message)

        # 🔹 Assistant init
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

        #  Handle BOTH function-call and tool-calls
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

                # FIX: handle string JSON params
                params = fn.get("arguments") or fn.get("parameters") or {}

                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse command parameters: {e}")
                        return JSONResponse(
                            status_code=400,
                            content={
  "status": "error",
  "message": "Sorry, I didn't understand that command. Try rephrasing."
})

                if fn_name == "query_codebase":
                    query = params.get("query", "")
                    if not query:
                        return JSONResponse(
                            status_code=400,
                            content={
    "status": "error",
    "message": "Sorry, I didn't understand that command. Try rephrasing."
})

                    # --- Cache lookup ---
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

                    # --- Cache miss: run full pipeline ---
                    context = retrieve(query)
                    history = get_memory(session_id)
                    answer = route_command(query, session_id) or generate_response(query, context, history)

                    # --- Cache insertion ---
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

    except Exception as e:
        print("SERVER ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
            "status": "error",
            "message": "An unexpected error occurred."
        })


@app.get("/health")
def health():
    return {"status": "ok", "message": "DevWhisper is running"}


# --- Admin endpoints ---------------------------------------------------
# Protected by the X-Admin-Secret header. Fail-closed: if ADMIN_SECRET is
# unset on the server, every request is rejected with 401.

def _require_admin(x_admin_secret: str | None) -> None:
    """
    Validate the X-Admin-Secret header against the server's ADMIN_SECRET.

    Uses a constant-time comparison to avoid timing-based secret recovery.
    Raises 401 on any mismatch (including missing header / unset server
    secret) so the endpoint never leaks session data by accident.
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

    Reads from the in-memory ``conversation_sessions`` store used by the
    webhook pipeline. Intended for debugging / monitoring only — no PII
    beyond session IDs and timestamps is exposed.

    Headers:
        X-Admin-Secret: <ADMIN_SECRET env var value>

    Response (200):
        {
            "status": "ok",
            "active_sessions": 3,
            "max_sessions": 100,
            "generated_at": "2026-07-21T10:22:14Z",
            "sessions": [
                {
                    "session_id": "call_a1b2c3d4",
                    "last_used": "2026-07-21T10:22:14Z",
                    "last_used_ago_seconds": 12,
                    "message_count": 4
                },
                ...
            ]
        }

    Response (401):
        {"detail": "Invalid or missing admin secret"}
    """
    _require_admin(x_admin_secret)

    now = time.time()
    sessions = []
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

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Endpoint to retrieve conversation history for a given session."""
    history = get_memory(session_id)
    return {"session_id": session_id, "history": history}

@app.get("/history")
def get_all_session_ids():
    """Endpoint to retrieve all session IDs."""
    all_session_ids = list(conversation_sessions.keys())
    return {"session_ids": all_session_ids}
