# 🎙️ DevWhisper — Voice-Native Developer Experience Agent

![License](https://img.shields.io/github/license/Aharshi3614/Devwhisper)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
![Open Issues](https://img.shields.io/github/issues/Aharshi3614/Devwhisper)
<!-- Add a CI build badge once a workflow exists, e.g.: -->
<!-- ![Build](https://img.shields.io/github/actions/workflow/status/Aharshi3614/Devwhisper/ci.yml) -->

DevWhisper is a voice-first AI agent built for developers. Instead of stopping to search through files or documentation, you just ask out loud — and it answers based on your actual codebase.

---

## 🚨 The Problem

Developers lose focus constantly. Switching between your editor, a browser, Stack Overflow, and documentation breaks the flow of thinking. Most AI tools still require you to type, copy-paste code, and wait.

DevWhisper lets you stay in flow. Ask a question with your voice, get an answer in seconds, and keep coding.

---

## ✨ What It Does

🎤 You ask a question about your code
🔍 It searches your actual codebase with hybrid vector + keyword search
🔊 It responds in plain spoken English, like a senior dev sitting next to you

Example questions that work:
- "What does the preprocess function do?"
- "Where is the model saved after training?"
- "How do I debug a KeyError in the pipeline?"

---

## 🏗️ Architecture

```
Developer speaks
      ↓
Vapi — Speech to Text
      ↓
FastAPI Webhook Server
      ↓
Qdrant + BM25 Hybrid Search
      ↓
Groq LLaMA 3.3 70B
      ↓
FastAPI sends answer back
      ↓
Vapi — Text to Speech
      ↓
Developer hears the response
```

---

## 🛠️ Tech Stack

| Component | Role |
|---|---|
| 🎙️ [Vapi](https://vapi.ai/) | Handles voice input and output |
| 🗄️ [Qdrant](https://qdrant.tech/) + BM25 | Hybrid vector + keyword search |
| 🤖 [Groq](https://groq.com/) (LLaMA 3.3 70B) | Generates the response |
| ⚡ [FastAPI](https://fastapi.tiangolo.com/) | Receives webhooks from Vapi and orchestrates everything |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for:
  - [Groq](https://console.groq.com/)
  - [Vapi](https://vapi.ai/)
  - [Qdrant](https://cloud.qdrant.io/) (cluster URL + API key)
- [ngrok](https://ngrok.com/) (or similar tunneling tool) to expose your local server to Vapi

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aharshi3614/Devwhisper.git
   cd Devwhisper
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example file and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Add your codebase**

   Drop the Python files you want DevWhisper to answer questions about into the `sample_codebase/` folder.

6. **Index your codebase**
   ```bash
   python indexer.py
   ```

> **💡 .gitignore awareness:** DevWhisper automatically skips files and
> directories listed in the codebase's `.gitignore` when indexing. Nested
> `.gitignore` files are honored too, so build artifacts, virtual
> environments, and other ignored files stay out of the search index.

7. **Start the server**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
3. Create a `.env` file in the root folder:`cp ./.env.example ./.env`

8. **Expose it publicly**
   ```bash
   ngrok http 8000
   ```

9. **Connect Vapi**

   Update your Vapi tool's Server URL to your ngrok URL plus `/webhook`.

### ✅ Verify it's working

With the server running, confirm it responds:
```bash
curl http://localhost:8000/health
```
You should get a response back confirming the server is live. You can also run the standalone test client (see below) to check the full pipeline end-to-end without needing a live voice call.

---

# 📋 Skipped File Reporting

During indexing, DevWhisper scans the codebase and filters files based on extension, size, and `.gitignore` rules. Files that are not indexed are reported with a specific reason so you can see exactly what was excluded and why.

## Skip Reasons

| Reason | When it applies | `detail` field |
| --- | --- | --- |
| `unsupported_extension` | File extension is not in `SUPPORTED_EXTENSIONS` (`.py`, `.md`) | The actual extension (e.g., `.txt`, `.png`) |
| `oversized` | File exceeds `MAX_FILE_SIZE_MB` (default 1 MB) | e.g., `"2.50 MB exceeds 1.00 MB limit"` |
| `gitignored` | File matches a `.gitignore` rule | `"matched by .gitignore rule"` |
| `unreadable` | `os.path.getsize()` raised an `OSError` (permissions, broken symlink) | The OS error message |

---

## Where to See Skip Reports

### 1. Server Logs

Every skipped file is logged at `INFO` level:

```text
Skipping unsupported file /path/to/notes.txt (extension '.txt' not in ['.md', '.py'])
Skipping oversized file /path/to/big.py (2.50 MB exceeds 1.00 MB limit)
Skipping gitignored file /path/to/secret.py

```

### 2. `/index/progress` SSE Stream

The `progress_state` payload includes skipped file details:

```json
{
  "skipped": [
    {
      "path": "/.../notes.txt",
      "size_bytes": null,
      "reason": "unsupported_extension",
      "detail": ".txt"
    },
    {
      "path": "/.../big.py",
      "size_bytes": 2621440,
      "reason": "oversized",
      "detail": "2.50 MB exceeds 1.00 MB limit"
    }
  ],
  "skipped_count": 2
}

```

### 3. `.index_cache.json`

The `_metadata.skipped_files` array records all skips from the most recent indexing run, persisted for offline inspection.

---

## Adding Support for a New File Type

Edit `SUPPORTED_EXTENSIONS` in `config.py`:

```python
SUPPORTED_EXTENSIONS: Final = frozenset({".py", ".md", ".txt"})

```

*After updating this configuration, re-index the project so files with the new extension are processed.*

---

## Architecture Documentation Snippet (`docx/architecture.md`)

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**SKIP REPORTING (Issue #223)**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

During indexing, files that are not indexed are reported with a reason
rather than silently dropped. The `collect_indexable_files()` function
in `indexer.py` returns a `skipped` list of dicts, each containing:

  - `path`       : Absolute path to the skipped file
  - `size_bytes` : File size in bytes (or `None` if unreadable)
  - `reason`     : One of `unsupported_extension`, `oversized`,
                   `gitignored`, `unreadable`
  - `detail`     : Human-readable context (the extension, the size
                   limit violation, the error message, etc.)

Every skip is also logged at `INFO` level via the application logger,
and the full `skipped` list is surfaced through the `/index/progress`
SSE stream and persisted in `.index_cache.json` under
`_metadata.skipped_files`.

```

## 🐳 Run with Docker

1. Build the image:
   ```bash
   docker build -t devwhisper .
   ```
2. Create a `.env` file with your API keys (same as above).
3. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env devwhisper
   ```
4. Expose it with ngrok and update your Vapi tool's Server URL as in the steps above.

---

## 🧪 Testing Locally

You can test DevWhisper's conversation flow directly from your terminal — without using Vapi or making a voice call — using the standalone test client. This is useful for local development and debugging.

1. Make sure your FastAPI server is running:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
2. Run the test client in interactive mode:
   ```bash
   python test_client.py
   ```
3. Or pass a one-off query:
   ```bash
   python test_client.py --query "What does the preprocess function do?"
   ```

---

## 📁 Project Structure

| File / Folder | Purpose |
|---|---|
| `main.py` | FastAPI webhook server, handles all Vapi events |
| `indexer.py` | Chunks, embeds, and uploads to Qdrant plus builds BM25 keyword index |
| `retriever.py` | Hybrid search: vector + BM25 + symbol matching fused via RRF |
| `llm.py` | Sends the query and context to Groq and returns the answer |
| `test_client.py` | Standalone CLI client for testing without Vapi |
| `sample_codebase/` | Put your own Python project files here |

---

## ⚙️ Configurable Recording Timeout

The maximum voice recording duration is configurable via the **Settings panel** (⚙️ icon, top-right).

- Default: **30 seconds**
- Range: **5 – 120 seconds**
- The setting is saved in `localStorage` and persists across sessions.
- While recording, a **live countdown** is displayed on the mic button showing remaining seconds.
- Recording **stops automatically** when the timeout expires and submits whatever was captured.

---

## ⚠️ Notes

- The ngrok URL changes every time you restart it — remember to update the Server URL in your Vapi tool settings each time.
- The `.env` file is **not** included in this repo for security. Create your own using `.env.example` as a starting point.

---

## 🤝 Contributing

Contributions are welcome! Check out the [open issues](https://github.com/Aharshi3614/Devwhisper/issues) — issues labeled `good first issue` are a great place to start.

## 📄 License

See the [LICENSE](./LICENSE) file for details.
