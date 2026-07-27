"""Central configuration for DevWhisper.

Non-secret application defaults live here so model, Qdrant, indexing, and LLM
settings are maintained in one place. Secrets remain in environment variables.
"""

import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a clear validation error."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a clear validation error."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


# Environment variable names
QDRANT_URL_ENV: Final = "QDRANT_URL"
QDRANT_API_KEY_ENV: Final = "QDRANT_API_KEY"
GROQ_API_KEY_ENV: Final = "GROQ_API_KEY"
LLM_API_KEY_ENV: Final = "LLM_API_KEY"
LLM_BASE_URL_ENV: Final = "LLM_BASE_URL"
LLM_MODEL_ENV: Final = "LLM_MODEL"

# Embedding and retrieval settings
EMBEDDING_MODEL_NAME: Final = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSIONS: Final = _env_int("EMBEDDING_DIMENSIONS", 384)
EMBEDDING_VERSION: Final = os.getenv("EMBEDDING_VERSION", "v1")
QDRANT_COLLECTION_NAME: Final = os.getenv(
    "QDRANT_COLLECTION_NAME", "devwhisper"
)
QDRANT_SIMILARITY_THRESHOLD: Final = _env_float(
    "QDRANT_SIMILARITY_THRESHOLD", 0.0
)
RETRIEVAL_TOP_K: Final = _env_int("RETRIEVAL_TOP_K", 6)

# Indexing settings
INDEX_CHUNK_SIZE: Final = _env_int("INDEX_CHUNK_SIZE", 15)
INDEX_CHUNK_OVERLAP: Final = _env_int("INDEX_CHUNK_OVERLAP", 3)
SUPPORTED_EXTENSIONS: Final = frozenset({".py" ,".md"})
SAMPLE_CODEBASE_DIRECTORY: Final = os.getenv(
    "SAMPLE_CODEBASE_DIRECTORY", "./sample_codebase"
)

# OpenAI-compatible LLM settings
DEFAULT_LLM_BASE_URL: Final = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL: Final = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_COMPATIBLE_MODEL: Final = "deepseek-v4-flash"
