"""
Unit tests for modular prompt construction pipeline (Issue #220).
"""

from prompt_builder import (
    prepare_context,
    prepare_user_content,
    build_messages,
    SYSTEM_PROMPT,
    USER_INSTRUCTIONS,
)
import llm


def test_prepare_context():
    raw_ctx = "  def foo(): pass  "
    processed = prepare_context(raw_ctx)
    assert processed == "def foo(): pass"

    truncated = prepare_context("1234567890", max_length=5)
    assert "...[context truncated]" in truncated
    assert truncated.startswith("12345")


def test_prepare_user_content():
    query = "What is foo?"
    context = "def foo(): pass"
    history = "User: Hi"

    content = prepare_user_content(query, context, history)
    assert "User question:\nWhat is foo?" in content
    assert "Code context:\ndef foo(): pass" in content
    assert "Conversation history:\nUser: Hi" in content
    assert USER_INSTRUCTIONS in content


def test_build_messages_structure():
    messages = build_messages("How to run?", "main.py content", "Previous conv")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT.strip()
    assert messages[1]["role"] == "user"
    assert "How to run?" in messages[1]["content"]


def test_llm_module_prompt_compatibility():
    """Verify that llm._SYSTEM_PROMPT and _USER_INSTRUCTIONS remain intact."""
    assert llm._SYSTEM_PROMPT == SYSTEM_PROMPT
    assert llm._USER_INSTRUCTIONS == USER_INSTRUCTIONS
