"""context_packer.py — Adaptive code context compression and token budgeting.

This module optimizes retrieved codebase context before injection into LLM prompts.
It provides token estimation, whitespace and comment compression, and priority-aware chunk
packing to prevent context window overflow while maximizing relevant code retention.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def estimate_token_count(text: str) -> int:
    """Estimate the number of LLM tokens in a string.

    Uses a fast character and word-boundary heuristic calibrated for code:
    approximately 1 token per 3.7 characters for code with punctuation.
    """
    if not text:
        return 0
    # Words + punctuation tokens
    words = len(re.findall(r"\w+|[^\w\s]", text))
    char_estimate = len(text) / 3.7
    return max(1, int((words * 0.5) + (char_estimate * 0.5)))


def compress_code_chunk(code: str, aggressive: bool = False) -> str:
    """Compress source code text by trimming redundant whitespace and empty comments.

    Args:
        code: Raw code chunk string.
        aggressive: If True, strips non-essential single-line comments.

    Returns:
        Cleaned, compressed code preserving indentation and logical semantics.
    """
    if not code:
        return ""

    lines = code.split("\n")
    compressed_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Remove empty lines excess
        if not stripped:
            if compressed_lines and compressed_lines[-1] != "":
                compressed_lines.append("")
            continue

        if aggressive:
            # Strip standalone single-line comments in aggressive mode
            if stripped.startswith(("#", "//")) and not stripped.startswith(("#!", "///")):
                continue

        # Trim trailing whitespace
        compressed_lines.append(line.rstrip())

    # Collapse multiple consecutive blank lines
    result = "\n".join(compressed_lines).strip()
    return result


def pack_context(
    context: str,
    max_tokens: int = 3000,
    enable_compression: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Pack retrieved context chunks into a budget-constrained context string.

    Args:
        context: Raw concatenated retrieval results (e.g. "Result 1: ... Result 2: ...").
        max_tokens: Maximum allowed token budget for the context block.
        enable_compression: Whether to apply whitespace/comment compression.

    Returns:
        Tuple of (packed_context_string, telemetry_dict).
    """
    if not context or not context.strip():
        return "", {
            "original_tokens": 0,
            "packed_tokens": 0,
            "compression_ratio": 1.0,
            "chunks_included": 0,
            "chunks_truncated": 0,
        }

    raw_tokens = estimate_token_count(context)
    if raw_tokens <= max_tokens and not enable_compression:
        return context, {
            "original_tokens": raw_tokens,
            "packed_tokens": raw_tokens,
            "compression_ratio": 1.0,
            "chunks_included": 1,
            "chunks_truncated": 0,
        }

    # Split by standard "Result N:" headers
    chunk_pattern = re.compile(r"(?=Result \d+:)")
    raw_chunks = [c.strip() for c in chunk_pattern.split(context) if c.strip()]
    if not raw_chunks:
        raw_chunks = [context.strip()]

    packed_chunks: List[str] = []
    current_tokens = 0
    chunks_truncated = 0

    for chunk in raw_chunks:
        processed_chunk = compress_code_chunk(chunk, aggressive=(raw_tokens > max_tokens * 1.5)) if enable_compression else chunk
        chunk_tokens = estimate_token_count(processed_chunk)

        if current_tokens + chunk_tokens <= max_tokens:
            packed_chunks.append(processed_chunk)
            current_tokens += chunk_tokens
        else:
            # If nothing was packed yet, include a truncated version of the top chunk
            if not packed_chunks:
                char_limit = int(max_tokens * 3.5)
                truncated_chunk = processed_chunk[:char_limit] + "\n...[truncated to fit token budget]"
                packed_chunks.append(truncated_chunk)
                current_tokens = estimate_token_count(truncated_chunk)
            chunks_truncated += 1

    packed_text = "\n\n".join(packed_chunks)
    final_tokens = estimate_token_count(packed_text)
    ratio = (final_tokens / raw_tokens) if raw_tokens > 0 else 1.0

    telemetry = {
        "original_tokens": raw_tokens,
        "packed_tokens": final_tokens,
        "compression_ratio": round(ratio, 4),
        "chunks_included": len(packed_chunks),
        "chunks_truncated": chunks_truncated,
    }

    return packed_text, telemetry
