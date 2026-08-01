"""
config.py — Central configuration for DevWhisper.

This module consolidates all non-secret application configuration in one place.
Model, Qdrant, indexing, and LLM settings are maintained here. Secrets (API keys,
URLs) remain in environment variables and are resolved at import time.

Configuration hierarchy (highest priority first):
    1. Environment variables (os.getenv)
    2. config.json file in project root
    3. Hardcoded defaults in this module

Validation:
    Several settings are validated at import time (e.g., chunk size constraints,
    file size limits) to fail fast on misconfiguration.

Usage:
    from config import EMBEDDING_MODEL_NAME, QDRANT_URL, etc.
"""

import json
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load .env file before reading any environment variables
load_dotenv()


def _load_json_config() -> dict:
    """
    Load optional configuration overrides from config.json in the project root.

    Returns:
        Parsed JSON dict, or empty dict if the file is missing or unreadable.
    """
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
    """
    Read an integer configuration value.

    Priority: environment variable → config.json → default.

    Args:
        name: Configuration key name.
        default: Fallback integer value.

    Returns:
        Resolved integer value.

    Raises:
        ValueError: If the resolved value cannot be parsed as an integer.
    """
    raw = os.getenv(name) or _JSON_CFG.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Config Error: {name} must be an integer, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    """
    Read an integer environment variable with a clear validation error.

    Args:
        name: Environment variable name.
        default: Fallback integer value.

    Returns:
        Resolved integer value.

    Raises:
        ValueError: If the environment variable is set but not a valid integer.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Config Error: {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    """
    Read a float environment variable with a clear validation error.

    Args:
        name: Environment variable name.
        default: Fallback float value.

    Returns:
        Resolved float value.

    Raises:
        ValueError: If the environment variable is set but not a valid number.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Config Error: {name} must be a number, got {value!r}") from exc


# ═══════════════════════════════════════════════════════════════════════════
# Embedding and retrieval settings
# ═══════════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME: Final = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
"""Sentence-transformers model used for dense vector embeddings."""

EMBEDDING_DIMENSIONS: Final = _env_int("EMBEDDING_DIMENSIONS", 384)
"""Dimensionality of the embedding vectors (must match the model output)."""

EMBEDDING_VERSION: Final = os.getenv("EMBEDDING_VERSION", "v1")
"""Version tag for the embedding model — used to detect stale indexes."""

QDRANT_COLLECTION_NAME: Final = os.getenv(
    "QDRANT_COLLECTION_NAME", "devwhisper"
)
"""Name of the Qdrant collection storing code chunk embeddings."""

QDRANT_SIMILARITY_THRESHOLD: Final = _env_float(
    "QDRANT_SIMILARITY_THRESHOLD", 0.0
)
"""Minimum cosine similarity score for Qdrant vector search results (0.0 = no filter)."""

RETRIEVAL_TOP_K: Final = _env_int("RETRIEVAL_TOP_K", 6)
"""Number of top results to return to the LLM after hybrid fusion."""


# ═══════════════════════════════════════════════════════════════════════════
# Cache settings
# ═══════════════════════════════════════════════════════════════════════════
CACHE_SIMILARITY_THRESHOLD: Final = _env_float(
    "CACHE_SIMILARITY_THRESHOLD", 0.70
)
"""Jaccard similarity threshold for near-duplicate query cache hits (0.0–1.0)."""


# ═══════════════════════════════════════════════════════════════════════════
# Indexing settings
# ═══════════════════════════════════════════════════════════════════════════
INDEX_CHUNK_SIZE: Final = _env_or_json("INDEX_CHUNK_SIZE", 15)
"""Number of lines per code chunk during indexing."""

INDEX_CHUNK_OVERLAP: Final = _env_or_json("INDEX_CHUNK_OVERLAP", 3)
"""Number of overlapping lines between consecutive chunks."""

# Validate indexing configuration at import time
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
"""File extensions eligible for indexing."""

SAMPLE_CODEBASE_DIRECTORY: Final = os.getenv(
    "SAMPLE_CODEBASE_DIRECTORY", "./sample_codebase"
)
"""Directory containing the codebase to be indexed."""


# ═══════════════════════════════════════════════════════════════════════════
# Maximum file size for indexing
# ═══════════════════════════════════════════════════════════════════════════
MAX_FILE_SIZE_MB: Final = _env_or_json("MAX_FILE_SIZE_MB", 1)
"""Maximum file size in megabytes for indexing eligibility."""

if MAX_FILE_SIZE_MB < 1:
    raise ValueError(f"Config Error: MAX_FILE_SIZE_MB must be >= 1, got {MAX_FILE_SIZE_MB}")
MAX_FILE_SIZE_BYTES: Final = MAX_FILE_SIZE_MB * 1024 * 1024
"""Maximum file size in bytes (derived from MAX_FILE_SIZE_MB)."""


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid retrieval settings
# ═══════════════════════════════════════════════════════════════════════════
RRF_K: Final = 60
"""Reciprocal Rank Fusion constant — dampens influence of low-ranked results."""

HYBRID_TOP_K: Final = _env_int("HYBRID_TOP_K", 20)
"""Number of candidates retrieved per strategy before RRF fusion."""

BM25_INDEX_PATH: Final = ".bm_index.pkl"
"""Local file path for the serialized BM25 keyword index."""


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI-compatible LLM settings
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_LLM_BASE_URL: Final = "https://api.groq.com/openai/v1"
"""Default base URL for the Groq OpenAI-compatible API."""

DEFAULT_GROQ_MODEL: Final = "llama-3.3-70b-versatile"
"""Default model when using Groq as the LLM provider."""

DEFAULT_OPENAI_COMPATIBLE_MODEL: Final = "deepseek-v4-flash"
"""Default model when using a custom OpenAI-compatible provider."""


# ═══════════════════════════════════════════════════════════════════════════
# Resolved environment variable values
# ═══════════════════════════════════════════════════════════════════════════
QDRANT_URL: Final[str | None] = os.getenv("QDRANT_URL")
"""Qdrant cluster URL (required for vector storage)."""

QDRANT_API_KEY: Final[str | None] = os.getenv("QDRANT_API_KEY")
"""Qdrant API key for authenticated access."""

GROQ_API_KEY: Final[str | None] = os.getenv("GROQ_API_KEY")
"""Groq LLM API key (used when no custom LLM provider is configured)."""

LLM_API_KEY: Final[str | None] = os.getenv("LLM_API_KEY")
"""Custom LLM API key (overrides Groq when set)."""

LLM_BASE_URL: Final[str] = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
"""Custom LLM base URL (defaults to Groq if not set)."""

LLM_MODEL: Final[str | None] = os.getenv("LLM_MODEL")
"""Custom LLM model name (optional override)."""

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
