from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from retriever import retrieve, embedder, client as qdrant_client
from llm import generate_response
from cache import get as cache_get, put as cache_put
import json
import time

app = FastAPI()

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
                    answer = generate_response(query, context, history)

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