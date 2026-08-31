"""tests/test_context_packer.py — Unit and integration tests for context compression and token budgeting."""

import pytest
from context_packer import estimate_token_count, compress_code_chunk, pack_context
from prompt_builder import prepare_context, generate_prompt_preview, build_messages


def test_estimate_token_count():
    assert estimate_token_count("") == 0
    short_text = "def hello():\n    return 'world'"
    tokens = estimate_token_count(short_text)
    assert tokens > 0
    assert tokens < 20


def test_compress_code_chunk():
    raw_code = """
    def compute(a, b):
        # Calculation comment
        
        result = a + b
        
        
        return result
    """
    compressed = compress_code_chunk(raw_code, aggressive=False)
    assert "def compute(a, b):" in compressed
    assert "return result" in compressed
    # Redundant consecutive empty lines removed
    assert "\n\n\n" not in compressed

    # Aggressive mode strips single-line comments
    aggressive_compressed = compress_code_chunk(raw_code, aggressive=True)
    assert "# Calculation comment" not in aggressive_compressed
    assert "return result" in aggressive_compressed


def test_pack_context_within_budget():
    context = """Result 1:
File: app.py
def run():
    print("running")

Result 2:
File: server.py
def serve():
    print("serving")
"""
    packed, telemetry = pack_context(context, max_tokens=1000, enable_compression=True)
    assert "Result 1:" in packed
    assert "Result 2:" in packed
    assert telemetry["chunks_included"] == 2
    assert telemetry["chunks_truncated"] == 0
    assert telemetry["packed_tokens"] <= 1000


def test_pack_context_over_budget_truncation():
    # Long context with 5 result blocks
    chunks = [f"Result {i}:\nFile: mod_{i}.py\ndef func_{i}():\n    return {i} * 100\n" for i in range(1, 10)]
    context = "\n\n".join(chunks)

    # Tight budget: only allows ~1-2 chunks
    packed, telemetry = pack_context(context, max_tokens=30, enable_compression=True)
    assert telemetry["chunks_truncated"] > 0
    assert telemetry["chunks_included"] < len(chunks)
    assert telemetry["packed_tokens"] <= 35


def test_prompt_builder_integration():
    sample_context = "Result 1:\nFile: a.py\ndef a(): pass"
    prepared = prepare_context(sample_context, max_tokens=100)
    assert "Result 1:" in prepared

    preview = generate_prompt_preview("What does a() do?", sample_context)
    assert "user_query" in preview
    assert "context_telemetry" in preview
    assert preview["context_telemetry"]["chunks_included"] == 1

    messages = build_messages("Query", sample_context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
