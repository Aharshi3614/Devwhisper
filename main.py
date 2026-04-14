from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from retriever import retrieve, embedder
from llm import generate_response

app = FastAPI()

# 🔥 Simple memory (last 5 interactions)
conversation_history = []


@app.on_event("startup")
async def startup_event():
    embedder.encode("warmup query")
    print("Embedder warmed up and ready!")


def update_memory(user, assistant):
    global conversation_history
    conversation_history.append(f"User: {user}\nAssistant: {assistant}")
    if len(conversation_history) > 5:
        conversation_history.pop(0)


def get_memory():
    return "\n\n".join(conversation_history)


@app.post("/webhook")
async def vapi_webhook(request: Request):
    body = await request.json()
    message = body.get("message", {})
    msg_type = message.get("type", "")

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
                                "query": {
                                    "type": "string"
                                }
                            },
                            "required": ["query"]
                        }
                    }]
                },
                "voice": {"provider": "11labs", "voiceId": "burt"}
            }
        })

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
            params = fn.get("arguments", {}) or fn.get("parameters", {})

            if fn_name == "query_codebase":
                query = params.get("query", "")

                context = retrieve(query)
                history = get_memory()

                answer = generate_response(query, context, history)

                update_memory(query, answer)

                results.append({
                    "toolCallId": tool.get("id", "single"),
                    "result": answer
                })

        return JSONResponse({"results": results})

    return JSONResponse({"status": "ok"})


@app.get("/health")
def health():
    return {"status": "DevWhisper is running"}