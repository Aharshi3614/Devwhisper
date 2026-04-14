from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from retriever import retrieve, embedder
from llm import generate_response

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    embedder.encode("warmup query")
    print("Embedder warmed up and ready!")

@app.post("/webhook")
async def vapi_webhook(request: Request):
    body = await request.json()
    print("Incoming body:", body)
    message = body.get("message", {})
    msg_type = message.get("type", "")

    if msg_type == "assistant-request":
        return JSONResponse({
            "assistant": {
                "firstMessage": "Hey, DevWhisper here. Tell me what you are working on or what error you are seeing.",
                "model": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "functions": [{
                        "name": "query_codebase",
                        "description": "Search the developer codebase to explain code, find functions, or debug errors.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The developer spoken question or error"
                                }
                            },
                            "required": ["query"]
                        }
                    }]
                },
                "voice": {"provider": "11labs", "voiceId": "burt"}
            }
        })

    if msg_type == "function-call":
        fn = message.get("functionCall", {})
        fn_name = fn.get("name", "")
        params = fn.get("parameters", {})
        print(f"Function called: {fn_name}, params: {params}")
        if fn_name == "query_codebase":
            query = params.get("query", "")
            print(f"Query: {query}")
            context = retrieve(query)
            print(f"Context retrieved: {context[:200]}")
            answer = generate_response(query, context)
            print(f"Answer: {answer}")
            return JSONResponse({"result": answer})

    if msg_type == "tool-calls":
        tool_calls = message.get("toolCalls", [])
        results = []
        for tool in tool_calls:
            fn = tool.get("function", {})
            fn_name = fn.get("name", "")
            params = fn.get("arguments", {})
            print(f"Tool called: {fn_name}, params: {params}")
            if fn_name == "query_codebase":
                query = params.get("query", "")
                print(f"Query: {query}")
                context = retrieve(query)
                print(f"Context: {context[:200]}")
                answer = generate_response(query, context)
                print(f"Answer: {answer}")
                results.append({
                    "toolCallId": tool.get("id", ""),
                    "result": answer
                })
        return JSONResponse({"results": results})

    return JSONResponse({"status": "ok"})

@app.get("/health")
def health():
    return {"status": "DevWhisper is running"}