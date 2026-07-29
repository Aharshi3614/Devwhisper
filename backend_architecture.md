# Backend Architecture Notes

DevWhisper now uses a shared dependency layer so backend components are not
hard-wired to concrete clients at import time.

## Dependency Flow

- `dependencies.py` owns the shared factories for:
  - Qdrant clients
  - retrieval embedders
  - indexing embedders
  - LLM clients and model selection
- `main.py` resolves backend dependencies at runtime and passes the needed
  service objects into retrieval and LLM calls.
- `retriever.py`, `llm.py`, and `indexer.py` now accept injected dependency
  objects and only fall back to shared factories when no override is supplied.

## Why This Helps

- Components are loosely coupled.
- Tests can provide fakes without patching module globals.
- Startup and shutdown code no longer depends on direct instantiation inside
  business logic.
- The indexing and query paths still behave the same from the caller's point
  of view.

## Runtime Shape

1. The FastAPI app starts.
2. The backend dependency factory provides shared clients.
3. Requests resolve the needed services and call the retrieval or LLM helpers.
4. Offline indexing uses the same pattern, but with the indexing dependency set.

