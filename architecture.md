🏗️ **DevWhisper — System Architecture**



DevWhisper is built on four core layers that work together to deliver

a seamless voice-to-code experience.



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



**LAYER 1 — VOICE INTERFACE**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Component     : Vapi

Role          : Handles all voice input and output

Input         : Developer's spoken question

Output        : Spoken answer back to the developer

How           : Converts speech to text, triggers webhook,

&#x20;               receives response, converts text back to speech



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



**LAYER 2 — BACKEND ORCHESTRATOR**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Component     : FastAPI (main.py)

Role          : Central brain that connects all components

Input         : Webhook POST from Vapi with the query

Output        : JSON response with the spoken answer

How           : Receives the query, calls retriever,passes context to LLM, returns final answer



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



**LAYER 3 — KNOWLEDGE RETRIEVAL**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Component     : Qdrant + retriever.py

Role          : Stores and searches the developer's codebase

Input         : Query encoded as a 384-dimension vector

Output        : Top 3 most relevant code chunks

How           : Code files are chunked and embedded using sentence-transformers (all-MiniLM-L6-v2) and stored in Qdrant. At query time, the question is embedded and cosine similarity is used to find the closest code                     chunks.



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



**LAYER 4 — LANGUAGE MODEL**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Component     : Groq API + llm.py

Model         : LLaMA 3.3 70B Versatile

Role          : Reasons over the retrieved code and generates a spoken-friendly answer

Input         : Developer query + retrieved code chunks

Output        : Plain English answer under 4 sentences

How           : Prompted to act as a voice-native pair programmer, no markdown, no bullet points



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



**DATA FLOW**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1  - Developer speaks a question out loud

Step 2  - Vapi transcribes speech to text

Step 3  - Vapi sends POST request to FastAPI /webhook

Step 4  - FastAPI extracts the query from the request

Step 5  - retriever.py encodes the query as a vector

Step 6  - Qdrant returns top 3 matching code chunks

Step 7  - FastAPI sends query + chunks to Groq LLaMA

Step 8  - Groq returns a concise spoken-friendly answer

Step 9  - FastAPI sends the answer back to Vapi

Step 10 - Vapi speaks the answer to the developer



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



**OFFLINE INDEXING PIPELINE**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1  - Developer adds .py files to sample\_codebase/

Step 2  - indexer.py reads and chunks each file into 15-line overlapping segments

Step 3  - Each chunk is embedded using all-MiniLM-L6-v2

Step 4  - Vectors + metadata uploaded to Qdrant cloud

Step 5  - Codebase is now searchable by voice



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**EMBEDDING VERSIONING & MIGRATION**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Embedding Versioning Was Introduced
As embedding models or configurations evolve (e.g. dimensions, overlap, model name, chunk sizes), queries generated using one model/configuration will yield poor results if run against an index created with a different version. Version management prevents sub-optimal semantic search results by warning operators when their current server configurations do not match the indexed database version.

### How Version Mismatches Are Detected
When a repository is indexed, the current `EMBEDDING_VERSION` (defaulting to `"v1"`) is stored within the repository metadata inside the `.index_cache.json` file. When the repository is loaded/used (e.g., during query retrieval or when starting an indexing run), the system parses the stored version metadata and compares it against the currently active server configuration. If they differ, a non-blocking warning is logged.

### When Repositories Should Be Re-Indexed
A repository should be re-indexed when the embedding version in the configuration changes (e.g., updating `EMBEDDING_VERSION` in config or changing the underlying model/chunk size settings). Re-indexing updates the vector database and metadata cache to align with the new embedding configurations.

### How Legacy Repositories Are Handled
Legacy repositories created before version management do not contain embedding version fields. To maintain backward compatibility and prevent immediate forced migration, legacy repositories are gracefully accepted. They are treated as valid, allowing retrieval and incremental indexing to continue without blocking operations or raising mismatch warnings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
