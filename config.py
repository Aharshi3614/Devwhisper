"""Central configuration for DevWhisper.

Non-secret application defaults live here so model, Qdrant, indexing, and LLM
settings are maintained in one place. Secrets remain in environment variables.
"""

import json
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()


def _load_json_config() -> dict:
    """Load values from config.json (project root) if it exists."""
    cfg_path = Path(__file__).resolve().parent / "config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Warning: failed to parse {cfg_path}, ignoring")
    return {}


_JSON_CFG: Final = _load_json_config()


def _env_or_json(name: str, default: int) -> int:
    """Read from env var first, then config.json, then default."""
    raw = os.getenv(name) or _JSON_CFG.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Config Error: {name} must be an integer, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a clear validation error."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Config Error: {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a clear validation error."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Config Error: {name} must be a number, got {value!r}") from exc


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

# Cache settings
CACHE_SIMILARITY_THRESHOLD: Final = _env_float(
    "CACHE_SIMILARITY_THRESHOLD", 0.70
)

# Indexing settings
INDEX_CHUNK_SIZE: Final = _env_or_json("INDEX_CHUNK_SIZE", 15)
INDEX_CHUNK_OVERLAP: Final = _env_or_json("INDEX_CHUNK_OVERLAP", 3)

# Validate indexing configuration
if INDEX_CHUNK_SIZE < 1:
    raise ValueError(
        f"Config Error: INDEX_CHUNK_SIZE must be >= 1, got {INDEX_CHUNK_SIZE}"
    )
if INDEX_CHUNK_OVERLAP < 0:
    raise ValueError(
        f"Config Error: INDEX_CHUNK_OVERLAP must be >= 0, got {INDEX_CHUNK_OVERLAP}"
    )
if INDEX_CHUNK_SIZE <= INDEX_CHUNK_OVERLAP:
    raise ValueError(
        f"Config Error: INDEX_CHUNK_SIZE ({INDEX_CHUNK_SIZE}) must be greater than "
        f"INDEX_CHUNK_OVERLAP ({INDEX_CHUNK_OVERLAP})"
    )

SUPPORTED_EXTENSIONS: Final = frozenset({".py", ".md"})
SAMPLE_CODEBASE_DIRECTORY: Final = os.getenv(
    "SAMPLE_CODEBASE_DIRECTORY", "./sample_codebase"
)

# Maximum file size for indexing
MAX_FILE_SIZE_MB: Final = _env_or_json("MAX_FILE_SIZE_MB", 1)
if MAX_FILE_SIZE_MB < 1:
    raise ValueError(f"Config Error: MAX_FILE_SIZE_MB must be >= 1, got {MAX_FILE_SIZE_MB}")
MAX_FILE_SIZE_BYTES: Final = MAX_FILE_SIZE_MB * 1024 * 1024

# Hybrid retrieval settings
RRF_K: Final = 60
HYBRID_TOP_K: Final = _env_int("HYBRID_TOP_K", 20)
BM25_INDEX_PATH: Final = ".bm_index.pkl"

# OpenAI-compatible LLM settings
DEFAULT_LLM_BASE_URL: Final = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL: Final = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_COMPATIBLE_MODEL: Final = "deepseek-v4-flash"

# Resolved environment variable values
QDRANT_URL: Final[str | None] = os.getenv("QDRANT_URL")
QDRANT_API_KEY: Final[str | None] = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY: Final[str | None] = os.getenv("GROQ_API_KEY")
LLM_API_KEY: Final[str | None] = os.getenv("LLM_API_KEY")
LLM_BASE_URL: Final[str] = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
LLM_MODEL: Final[str | None] = os.getenv("LLM_MODEL")


def validate_config() -> None:
    """Validate runtime configuration values and print actionable errors if invalid."""
    errors = []

    if RETRIEVAL_TOP_K < 1:
        errors.append(f"RETRIEVAL_TOP_K must be at least 1, got {RETRIEVAL_TOP_K}")

    if HYBRID_TOP_K < 1:
        errors.append(f"HYBRID_TOP_K must be at least 1, got {HYBRID_TOP_K}")

    if CACHE_SIMILARITY_THRESHOLD < 0.0 or CACHE_SIMILARITY_THRESHOLD > 1.0:
        errors.append(
            f"CACHE_SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {CACHE_SIMILARITY_THRESHOLD}"
        )

    if errors:
        error_msg = "Invalid configuration detected:\n" + "\n".join(f"- {e}" for s in errors for e in [s])
        raise ValueError(error_msg)


# Run validation on import
validate_config()

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_VERSION",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_SIMILARITY_THRESHOLD",
    "RETRIEVAL_TOP_K",
    "CACHE_SIMILARITY_THRESHOLD",
    "INDEX_CHUNK_SIZE",
    "INDEX_CHUNK_OVERLAP",
    "SUPPORTED_EXTENSIONS",
    "SAMPLE_CODEBASE_DIRECTORY",
    "MAX_FILE_SIZE_MB",
    "MAX_FILE_SIZE_BYTES",
    "RRF_K",
    "HYBRID_TOP_K",
    "BM25_INDEX_PATH",
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_OPENAI_COMPATIBLE_MODEL",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "GROQ_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "validate_config",
]
