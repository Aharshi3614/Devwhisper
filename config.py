"""config.py — Central configuration for DevWhisper.

This module consolidates all non-secret application configuration in one place.
Model, Qdrant, indexing, and LLM settings are maintained here. Secrets (API keys,
URLs) remain in environment variables and are resolved at import time.

Configuration hierarchy (highest priority first):
 1. Environment variables (os.getenv)
 2. config.json file in project root
 3. Hardcoded defaults in this module

Validation (issue #226 — improved error messages):
 Several settings are validated at import time. When validation fails, a
 :class:`ConfigError` is raised with a descriptive message that:
   - Clearly identifies the offending setting by name
   - Shows the invalid value that was provided
   - States the expected type / range / format
   - Suggests a concrete fix (env var name, example value, or doc link)

 All error messages use a consistent format::

     ConfigError: <SETTING_NAME> — <what was wrong>
                   Got: <value>
                   Expected: <type/range>
                   Fix: <actionable suggestion>

Usage:
 from config import EMBEDDING_MODEL_NAME, QDRANT_URL, etc.
"""

import json
import os
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

# Load .env file before reading any environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# Issue #226: ConfigError with actionable diagnostics
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when a configuration value is invalid.

    Subclass of :class:`ValueError` so existing ``except ValueError``
    handlers continue to work, while providing richer diagnostics via
    :attr:`setting`, :attr:`value`, :attr:`expected`, and :attr:`fix`.

    Attributes:
        setting:  The name of the misconfigured setting (env var / JSON key).
        value:    The invalid value that was provided.
        expected: Human-readable description of what was expected
                  (type, range, or format).
        fix:      Concrete, actionable suggestion for resolving the error.
    """

    def __init__(
        self,
        setting: str,
        value: Any,
        expected: str,
        fix: str,
    ) -> None:
        self.setting = setting
        self.value = value
        self.expected = expected
        self.fix = fix
        message = (
            f"{setting} — invalid configuration.\n"
            f"  Got:      {value!r}\n"
            f"  Expected: {expected}\n"
            f"  Fix:      {fix}"
        )
        super().__init__(message)


def _format_fix_env(env_var: str, example: str) -> str:
    """Build a standard 'set the env var' fix suggestion."""
    return (
        f"Set the {env_var} environment variable to a valid value "
        f"(e.g. `export {env_var}={example}`), or remove it to use the default. "
        f"You can also set it in config.json or your .env file."
    )


def _format_fix_json(json_key: str, example: str) -> str:
    """Build a standard 'fix in config.json' suggestion."""
    return (
        f"Edit config.json and set \"{json_key}\" to a valid value "
        f"(e.g. \"{json_key}\": {example}), or remove the key to use the default."
    )


# ---------------------------------------------------------------------------
# JSON config loader
# ---------------------------------------------------------------------------

def _load_json_config() -> dict:
    """
    Load optional configuration overrides from config.json in the project root.

    Returns:
        Parsed JSON dict, or empty dict if the file is missing or unreadable.

    Note:
        If the file exists but is invalid JSON, a clear warning is printed
        so the user knows their overrides are being ignored (issue #226).
    """
    cfg_path = Path(__file__).resolve().parent / "config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            print(
                f"Warning: failed to parse {cfg_path} — {exc}. "
                f"Configuration overrides in this file will be ignored. "
                f"Fix the JSON syntax (e.g. trailing commas, missing quotes) "
                f"and restart."
            )
        except OSError as exc:
            print(
                f"Warning: could not read {cfg_path} — {exc}. "
                f"Configuration overrides in this file will be ignored."
            )
    return {}


_JSON_CFG: Final = _load_json_config()


# ---------------------------------------------------------------------------
# Typed env / JSON accessors with rich errors
# ---------------------------------------------------------------------------

def _env_or_json_int(name: str, default: int, *, min_value: int | None = None) -> int:
    """
    Read an integer configuration value from env var or config.json.

    Priority: environment variable → config.json → default.

    Args:
        name:      Configuration key name (used for both env var and JSON key).
        default:   Fallback integer value.
        min_value: Optional inclusive lower bound. If the resolved value is
                   below this, a :class:`ConfigError` is raised with a fix
                   suggestion.

    Returns:
        Resolved integer value.

    Raises:
        ConfigError: If the value cannot be parsed as an integer, or is
                     below ``min_value``.
    """
    raw = os.getenv(name) or _JSON_CFG.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        source = "environment variable" if os.getenv(name) is not None else "config.json"
        raise ConfigError(
            setting=name,
            value=raw,
            expected="an integer (e.g. 15, 20, 50)",
            fix=(
                f"The {source} {name}={raw!r} is not a valid integer. "
                + (_format_fix_env(name, "15") if source == "environment variable"
                   else _format_fix_json(name, "15"))
            ),
        ) from None

    if min_value is not None and value < min_value:
        raise ConfigError(
            setting=name,
            value=value,
            expected=f"an integer >= {min_value}",
            fix=(
                f"{name}={value} is below the minimum allowed value ({min_value}). "
                + (_format_fix_env(name, str(max(min_value, default))) if os.getenv(name) is not None
                   else _format_fix_json(name, str(max(min_value, default))))
            ),
        )
    return value


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    """
    Read an integer environment variable with a clear validation error.

    Args:
        name:      Environment variable name.
        default:   Fallback integer value.
        min_value: Optional inclusive lower bound.

    Returns:
        Resolved integer value.

    Raises:
        ConfigError: If the env var is set but not a valid integer, or is
                     below ``min_value``.
    """
    value_str = os.getenv(name)
    if value_str is None:
        return default
    try:
        value = int(value_str)
    except ValueError:
        raise ConfigError(
            setting=name,
            value=value_str,
            expected="an integer (e.g. 6, 20, 100)",
            fix=_format_fix_env(name, "6"),
        ) from None

    if min_value is not None and value < min_value:
        raise ConfigError(
            setting=name,
            value=value,
            expected=f"an integer >= {min_value}",
            fix=(
                f"{name}={value} is below the minimum allowed value ({min_value}). "
                + _format_fix_env(name, str(max(min_value, default)))
            ),
        )
    return value


def _env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """
    Read a float environment variable with a clear validation error.

    Args:
        name:      Environment variable name.
        default:   Fallback float value.
        min_value: Optional inclusive lower bound.
        max_value: Optional inclusive upper bound.

    Returns:
        Resolved float value.

    Raises:
        ConfigError: If the env var is set but not a valid number, or is
                     outside the ``[min_value, max_value]`` range.
    """
    value_str = os.getenv(name)
    if value_str is None:
        return default
    try:
        value = float(value_str)
    except ValueError:
        raise ConfigError(
            setting=name,
            value=value_str,
            expected="a number (e.g. 0.0, 0.75, 1.0)",
            fix=_format_fix_env(name, "0.75"),
        ) from None

    if min_value is not None and value < min_value:
        raise ConfigError(
            setting=name,
            value=value,
            expected=f"a number >= {min_value}",
            fix=(
                f"{name}={value} is below the minimum allowed value ({min_value}). "
                + _format_fix_env(name, str(default))
            ),
        )
    if max_value is not None and value > max_value:
        raise ConfigError(
            setting=name,
            value=value,
            expected=f"a number <= {max_value}",
            fix=(
                f"{name}={value} is above the maximum allowed value ({max_value}). "
                + _format_fix_env(name, str(default))
            ),
        )
    return value


def _env_str(name: str, default: str, *, allowed: frozenset[str] | None = None) -> str:
    """
    Read a string environment variable with optional allowed-values validation.

    Args:
        name:     Environment variable name.
        default:  Fallback string value.
        allowed:  Optional set of allowed values. If set and the resolved
                  value is not in the set, a :class:`ConfigError` is raised.

    Returns:
        Resolved string value.

    Raises:
        ConfigError: If ``allowed`` is set and the value is not in it.
    """
    value = os.getenv(name, default)
    if allowed is not None and value not in allowed:
        allowed_str = ", ".join(repr(v) for v in sorted(allowed))
        raise ConfigError(
            setting=name,
            value=value,
            expected=f"one of: {allowed_str}",
            fix=_format_fix_env(name, next(iter(sorted(allowed)))),
        )
    return value


# ---------------------------------------------------------------------------
# Backward-compatible aliases (existing callers use the old names)
# ---------------------------------------------------------------------------
_env_or_json = _env_or_json_int


# ═══════════════════════════════════════════════════════════════════════════
# Embedding and retrieval settings
# ═══════════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME: Final = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
"""Sentence-transformers model used for dense vector embeddings."""

EMBEDDING_DIMENSIONS: Final = _env_int("EMBEDDING_DIMENSIONS", 384, min_value=1)
"""Dimensionality of the embedding vectors (must match the model output)."""

EMBEDDING_VERSION: Final = os.getenv("EMBEDDING_VERSION", "v2")
"""Version tag for the embedding model — used to detect stale indexes."""

QDRANT_COLLECTION_NAME: Final = os.getenv(
    "QDRANT_COLLECTION_NAME", "devwhisper"
)
"""Name of the Qdrant collection storing code chunk embeddings."""

QDRANT_SIMILARITY_THRESHOLD: Final = _env_float(
    "QDRANT_SIMILARITY_THRESHOLD", 0.0, min_value=0.0, max_value=1.0
)
"""Minimum cosine similarity score for Qdrant vector search results (0.0 = no filter)."""

RETRIEVAL_TOP_K: Final = _env_int("RETRIEVAL_TOP_K", 6, min_value=1)
"""Number of top results to return to the LLM after hybrid fusion."""


# ═══════════════════════════════════════════════════════════════════════════
# Prompt budget
# ═══════════════════════════════════════════════════════════════════════════
MAX_PROMPT_CONTEXT_CHARS: Final = _env_int(
    "MAX_PROMPT_CONTEXT_CHARS", 48_000, min_value=1_000
)
"""Character ceiling for the retrieved code context in one prompt.

Nothing upstream bounds this. A chunk is INDEX_CHUNK_SIZE *lines*, and a line
has no length limit — minified JS, generated code, long data literals and
vendored files all pass collect_indexable_files() and are chunked by line
count. MAX_FILE_SIZE_MB bounds the file, not the chunk. So RETRIEVAL_TOP_K
chunks could exceed any provider's context window, and the request failed with
nothing but a generic apology to show for it.

48,000 characters is roughly 12–16k tokens of source, which leaves ample room
for the system prompt, history and the answer inside a modest window while
being generous enough that ordinary repositories never reach it.
"""

MAX_PROMPT_HISTORY_CHARS: Final = _env_int(
    "MAX_PROMPT_HISTORY_CHARS", 8_000, min_value=0
)
"""Character ceiling for conversation history in one prompt.

Budgeted separately from code context, and deliberately much smaller. History
holds up to max_history_per_session complete previous answers; sharing one
budget let an old, long answer crowd out the code the current question is
actually about, which is the less valuable of the two.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Cache settings
# ═══════════════════════════════════════════════════════════════════════════
CACHE_SIMILARITY_THRESHOLD: Final = _env_float(
    "CACHE_SIMILARITY_THRESHOLD", 0.70, min_value=0.0, max_value=1.0
)
"""Jaccard similarity threshold for near-duplicate query cache hits (0.0–1.0)."""


# ═══════════════════════════════════════════════════════════════════════════
# Indexing settings
# ═══════════════════════════════════════════════════════════════════════════
INDEX_CHUNK_SIZE: Final = _env_or_json("INDEX_CHUNK_SIZE", 15, min_value=1)
"""Number of lines per code chunk during indexing."""

INDEX_CHUNK_OVERLAP: Final = _env_or_json("INDEX_CHUNK_OVERLAP", 3, min_value=0)
"""Number of overlapping lines between consecutive chunks."""

# Validate indexing configuration at import time (issue #226: rich errors)
if INDEX_CHUNK_SIZE <= INDEX_CHUNK_OVERLAP:
    raise ConfigError(
        setting="INDEX_CHUNK_SIZE / INDEX_CHUNK_OVERLAP",
        value=f"chunk_size={INDEX_CHUNK_SIZE}, overlap={INDEX_CHUNK_OVERLAP}",
        expected="INDEX_CHUNK_SIZE > INDEX_CHUNK_OVERLAP (chunks must be larger than their overlap)",
        fix=(
            "Increase INDEX_CHUNK_SIZE or decrease INDEX_CHUNK_OVERLAP. "
            "For example, in config.json set "
            "\"INDEX_CHUNK_SIZE\": 15 and \"INDEX_CHUNK_OVERLAP\": 3, "
            "or export INDEX_CHUNK_SIZE=20 in your shell."
        ),
    )

SUPPORTED_EXTENSIONS: Final = frozenset(
    {".py", ".md", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}
)
"""File extensions eligible for indexing."""


SAMPLE_CODEBASE_DIRECTORY: Final = os.getenv(
    "SAMPLE_CODEBASE_DIRECTORY", "./sample_codebase"
)
"""Directory containing the codebase to be indexed."""


# ═══════════════════════════════════════════════════════════════════════════
# Maximum file size for indexing
# ═══════════════════════════════════════════════════════════════════════════
MAX_FILE_SIZE_MB: Final = _env_or_json("MAX_FILE_SIZE_MB", 1, min_value=1)
"""Maximum file size in megabytes for indexing eligibility."""

MAX_FILE_SIZE_BYTES: Final = MAX_FILE_SIZE_MB * 1024 * 1024
"""Maximum file size in bytes (derived from MAX_FILE_SIZE_MB)."""


# ═══════════════════════════════════════════════════════════════════════════
# Uploaded archive limits (issue #310)
# ═══════════════════════════════════════════════════════════════════════════
# MAX_FILE_SIZE_BYTES above bounds a *source file* the indexer chooses to read.
# It is applied by collect_indexable_files(), which runs after extraction has
# already written the bytes to disk, so it does not bound an upload. These do.
MAX_UPLOAD_SIZE_MB: Final = _env_or_json("MAX_UPLOAD_SIZE_MB", 100, min_value=1)
"""Maximum size of an uploaded ZIP archive, in megabytes."""

MAX_UPLOAD_SIZE_BYTES: Final = MAX_UPLOAD_SIZE_MB * 1024 * 1024
"""Maximum size of an uploaded ZIP archive, in bytes."""

MAX_EXTRACTED_SIZE_MB: Final = _env_or_json(
    "MAX_EXTRACTED_SIZE_MB", 500, min_value=1
)
"""Maximum total uncompressed size of an uploaded archive, in megabytes."""

MAX_EXTRACTED_SIZE_BYTES: Final = MAX_EXTRACTED_SIZE_MB * 1024 * 1024
"""Maximum total uncompressed size of an uploaded archive, in bytes."""

MAX_ARCHIVE_ENTRIES: Final = _env_or_json("MAX_ARCHIVE_ENTRIES", 20000, min_value=1)
"""Maximum number of members in an uploaded archive."""

MAX_COMPRESSION_RATIO: Final = _env_float(
    "MAX_COMPRESSION_RATIO", 100.0, min_value=1.0
)
"""Maximum uncompressed-to-compressed ratio for an uploaded archive.

Checked in addition to the absolute total because the two catch different
things: the total catches "this is simply too big", the ratio catches a small
upload engineered to expand enormously — the shape of a deliberate bomb rather
than of a large project. Ordinary source trees compress well under 20x.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid retrieval settings
# ═══════════════════════════════════════════════════════════════════════════
RRF_K: Final = 60
"""Reciprocal Rank Fusion constant — dampens influence of low-ranked results."""

HYBRID_TOP_K: Final = _env_int("HYBRID_TOP_K", 20, min_value=1)
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

OUTPUT_DIRECTORY: Final = "./output"
"""Directory storing each repository's index artifacts: BM25, cache, and registry."""


# ---------------------------------------------------------------------------
# Runtime validation (issue #226: collect ALL errors, not just the first)
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Validate runtime configuration values and raise :class:`ConfigError`
    if any are invalid.

    Unlike the import-time checks (which fail fast on the first error),
    this function collects ALL validation errors and reports them together
    so the user can fix everything in one pass instead of fix-restart-fix
    cycles.

    Raises:
        ConfigError: If any configuration value is invalid. The error
                     message lists every problem found, with suggested
                     fixes for each.
    """
    errors: list[str] = []

    if MAX_PROMPT_CONTEXT_CHARS < 1000:
        errors.append(
            f"- MAX_PROMPT_CONTEXT_CHARS must be at least 1000, got "
            f"{MAX_PROMPT_CONTEXT_CHARS}. A budget below that cannot hold a "
            f"single chunk. Fix: export MAX_PROMPT_CONTEXT_CHARS=48000"
        )

    if MAX_PROMPT_HISTORY_CHARS < 0:
        errors.append(
            f"- MAX_PROMPT_HISTORY_CHARS must be >= 0, got "
            f"{MAX_PROMPT_HISTORY_CHARS}. Use 0 to drop history entirely. "
            f"Fix: export MAX_PROMPT_HISTORY_CHARS=8000"
        )

    if RETRIEVAL_TOP_K < 1:
        errors.append(
            f"- RETRIEVAL_TOP_K must be at least 1, got {RETRIEVAL_TOP_K}. "
            f"Fix: export RETRIEVAL_TOP_K=6 (or set in .env)"
        )

    if HYBRID_TOP_K < 1:
        errors.append(
            f"- HYBRID_TOP_K must be at least 1, got {HYBRID_TOP_K}. "
            f"Fix: export HYBRID_TOP_K=20 (or set in .env)"
        )

    if CACHE_SIMILARITY_THRESHOLD < 0.0 or CACHE_SIMILARITY_THRESHOLD > 1.0:
        errors.append(
            f"- CACHE_SIMILARITY_THRESHOLD must be between 0.0 and 1.0, "
            f"got {CACHE_SIMILARITY_THRESHOLD}. "
            f"Fix: export CACHE_SIMILARITY_THRESHOLD=0.70 (or set in .env)"
        )

    if QDRANT_SIMILARITY_THRESHOLD < 0.0 or QDRANT_SIMILARITY_THRESHOLD > 1.0:
        errors.append(
            f"- QDRANT_SIMILARITY_THRESHOLD must be between 0.0 and 1.0, "
            f"got {QDRANT_SIMILARITY_THRESHOLD}. "
            f"Fix: export QDRANT_SIMILARITY_THRESHOLD=0.0 (or set in .env)"
        )

    if EMBEDDING_DIMENSIONS < 1:
        errors.append(
            f"- EMBEDDING_DIMENSIONS must be a positive integer, got {EMBEDDING_DIMENSIONS}. "
            f"Fix: export EMBEDDING_DIMENSIONS=384 (must match your model)"
        )

    if INDEX_CHUNK_SIZE < 1:
        errors.append(
            f"- INDEX_CHUNK_SIZE must be >= 1, got {INDEX_CHUNK_SIZE}. "
            f"Fix: set \"INDEX_CHUNK_SIZE\": 15 in config.json"
        )

    if INDEX_CHUNK_OVERLAP < 0:
        errors.append(
            f"- INDEX_CHUNK_OVERLAP must be >= 0, got {INDEX_CHUNK_OVERLAP}. "
            f"Fix: set \"INDEX_CHUNK_OVERLAP\": 3 in config.json"
        )

    if INDEX_CHUNK_SIZE <= INDEX_CHUNK_OVERLAP:
        errors.append(
            f"- INDEX_CHUNK_SIZE ({INDEX_CHUNK_SIZE}) must be greater than "
            f"INDEX_CHUNK_OVERLAP ({INDEX_CHUNK_OVERLAP}). "
            f"Fix: increase INDEX_CHUNK_SIZE or decrease INDEX_CHUNK_OVERLAP in config.json"
        )

    if MAX_FILE_SIZE_MB < 1:
        errors.append(
            f"- MAX_FILE_SIZE_MB must be >= 1, got {MAX_FILE_SIZE_MB}. "
            f"Fix: set \"MAX_FILE_SIZE_MB\": 1 in config.json"
        )

    # Required-for-production checks (warnings, not errors — startup
    # behavior must remain unchanged per issue #226 acceptance criteria).
    # These are surfaced as informational messages so users know what's
    # missing, but they do NOT raise.
    missing_secrets: list[str] = []
    if QDRANT_URL is None:
        missing_secrets.append("QDRANT_URL")
    if QDRANT_API_KEY is None:
        missing_secrets.append("QDRANT_API_KEY")
    if GROQ_API_KEY is None and LLM_API_KEY is None:
        missing_secrets.append("GROQ_API_KEY or LLM_API_KEY")

    if missing_secrets:
        # Print as a warning, not an error — startup continues.
        print(
            "Config Warning: the following environment variables are not set: "
            + ", ".join(missing_secrets)
            + ". Some features (vector search, LLM generation) will not work "
            "until they are added to your .env file. See .env.example."
        )

    if errors:
        error_msg = (
            "Invalid configuration detected — "
            f"{len(errors)} problem(s) found:\n\n"
            + "\n".join(errors)
            + "\n\nSee .env.example for the full list of supported settings."
        )
        raise ValueError(error_msg)


# Run validation on import
validate_config()


PROMPT_PREVIEW_MODE: bool = os.getenv("PROMPT_PREVIEW_MODE", "false").lower() in ("true", "1", "yes")

__all__ = [
    "ConfigError",
    "OUTPUT_DIRECTORY",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_VERSION",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_SIMILARITY_THRESHOLD",
    "RETRIEVAL_TOP_K",
    "MAX_PROMPT_CONTEXT_CHARS",
    "MAX_PROMPT_HISTORY_CHARS",
    "CACHE_SIMILARITY_THRESHOLD",
    "INDEX_CHUNK_SIZE",
    "INDEX_CHUNK_OVERLAP",
    "SUPPORTED_EXTENSIONS",
    "SAMPLE_CODEBASE_DIRECTORY",
    "MAX_FILE_SIZE_MB",
    "MAX_FILE_SIZE_BYTES",
    "MAX_UPLOAD_SIZE_MB",
    "MAX_UPLOAD_SIZE_BYTES",
    "MAX_EXTRACTED_SIZE_MB",
    "MAX_EXTRACTED_SIZE_BYTES",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_COMPRESSION_RATIO",
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
    "PROMPT_PREVIEW_MODE",
    "validate_config",
]
