"""Regression tests for the prompt context budget (issue #295).

`prepare_context()` has always taken a `max_length` and truncated when it was
set. Both callers omitted it, so it was always None, the branch was dead, and
the retrieved context went to the provider at whatever size retrieval produced.

Nothing upstream bounded that size. A chunk is INDEX_CHUNK_SIZE *lines* and a
line has no length limit, so minified JS, generated code, long data literals and
vendored files all produce chunks of arbitrary size — MAX_FILE_SIZE_MB bounds
the file, not the chunk. Six of those plus five previous turns of history
overran the context window, the provider rejected the request, and the user got
a generic apology that said nothing about prompt size.
"""

import pytest

from config import MAX_PROMPT_CONTEXT_CHARS, MAX_PROMPT_HISTORY_CHARS
from prompt_builder import (
    HISTORY_TRUNCATION_NOTICE,
    TRUNCATION_NOTICE,
    build_messages,
    generate_prompt_preview,
    prepare_context,
    prepare_history,
    prepare_user_content,
)


def _result_block(index, body_lines=3, filler="x" * 40):
    """One block in the shape retriever.retrieve() emits."""
    body = "\n".join(f"    {filler}" for _ in range(body_lines))
    return (
        f"Result {index}:\n"
        f"File: module_{index}.py\n"
        f"Function: handler_{index}\n"
        f"Location: Line {index * 10}\n"
        f"Code:\n{body}\n"
    )


def _context(count, **kwargs):
    return "\n\n".join(_result_block(i, **kwargs) for i in range(1, count + 1))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_budget_defaults_are_sane():
    assert MAX_PROMPT_CONTEXT_CHARS >= 1000
    assert MAX_PROMPT_HISTORY_CHARS >= 0
    # History must not be allowed to crowd out code context.
    assert MAX_PROMPT_HISTORY_CHARS < MAX_PROMPT_CONTEXT_CHARS


def test_default_is_the_configured_budget_not_unlimited():
    """The old default was None, which meant "never truncate"."""
    oversized = "y" * (MAX_PROMPT_CONTEXT_CHARS + 5000)
    prepared = prepare_context(oversized)

    assert len(prepared) < len(oversized)
    assert TRUNCATION_NOTICE in prepared


def test_context_within_budget_is_untouched():
    context = _context(3)
    assert prepare_context(context) == context.strip()
    assert TRUNCATION_NOTICE not in prepare_context(context)


def test_zero_disables_truncation_explicitly():
    oversized = "y" * (MAX_PROMPT_CONTEXT_CHARS + 5000)
    assert prepare_context(oversized, max_length=0) == oversized


def test_empty_context_stays_empty():
    assert prepare_context("") == ""
    assert prepare_context(None) == ""


# ---------------------------------------------------------------------------
# Boundary-aware truncation
# ---------------------------------------------------------------------------
def test_truncation_cuts_between_result_blocks():
    """A mid-block cut hands the model half an identifier and a bare Code: header."""
    context = _context(10)
    budget = len(_result_block(1)) * 3

    prepared = prepare_context(context, max_length=budget)
    body = prepared[: -len(TRUNCATION_NOTICE)]

    assert "Result 1:" in body
    # Whatever survived, the last block is complete: no dangling Code: header.
    assert not body.rstrip().endswith("Code:")
    # And no partial block header was left behind.
    assert body.count("Code:") == body.count("Result ")


def test_truncation_keeps_the_highest_ranked_results():
    """retrieve() orders best-first, so the prefix is the part worth keeping."""
    context = _context(10)
    prepared = prepare_context(context, max_length=len(_result_block(1)) * 3)

    assert "Result 1:" in prepared
    assert "Result 10:" not in prepared


def test_truncation_never_exceeds_the_budget_by_more_than_the_notice():
    context = _context(10)
    budget = 600
    prepared = prepare_context(context, max_length=budget)

    assert len(prepared) <= budget + len(TRUNCATION_NOTICE)


def test_a_single_oversized_result_still_yields_something():
    """One enormous chunk must not truncate the whole context to nothing."""
    context = _result_block(1, body_lines=200)
    prepared = prepare_context(context, max_length=400)

    assert prepared.strip()
    assert "Result 1:" in prepared
    assert TRUNCATION_NOTICE in prepared


def test_a_single_oversized_result_is_cut_at_a_line_boundary():
    context = _result_block(1, body_lines=200)
    prepared = prepare_context(context, max_length=1000)
    body = prepared[: -len(TRUNCATION_NOTICE)]

    # The final line the model sees is a whole one.
    assert body.split("\n")[-1].strip().endswith("x")


def test_context_without_result_markers_is_still_bounded():
    """Callers can pass arbitrary text; the budget must hold regardless."""
    prepared = prepare_context("z" * 5000, max_length=1000)
    assert len(prepared) <= 1000 + len(TRUNCATION_NOTICE)


def test_truncation_is_logged_with_both_sizes(caplog):
    with caplog.at_level("WARNING"):
        prepare_context(_context(10), max_length=600)

    assert "truncated" in caplog.text
    assert "RETRIEVAL_TOP_K" in caplog.text


def test_no_warning_when_nothing_is_truncated(caplog):
    with caplog.at_level("WARNING"):
        prepare_context(_context(2))

    assert "truncated" not in caplog.text


# ---------------------------------------------------------------------------
# History budget
# ---------------------------------------------------------------------------
def test_history_within_budget_is_untouched():
    history = "User: hi\nAssistant: hello"
    assert prepare_history(history) == history


def test_history_is_trimmed_to_its_own_budget():
    history = "\n\n".join(
        f"User: question {i}\nAssistant: {'answer ' * 200}" for i in range(10)
    )
    prepared = prepare_history(history)

    assert len(prepared) < len(history)
    assert len(prepared) <= MAX_PROMPT_HISTORY_CHARS + len(HISTORY_TRUNCATION_NOTICE)


def test_history_trimming_keeps_the_most_recent_turns():
    history = "\n\n".join(
        f"User: question {i}\nAssistant: {'filler ' * 40}" for i in range(10)
    )
    prepared = prepare_history(history, max_length=1200)

    assert "question 9" in prepared
    assert "question 0" not in prepared


def test_trimmed_history_starts_at_an_exchange_boundary():
    history = "\n\n".join(
        f"User: question {i}\nAssistant: {'filler ' * 40}" for i in range(10)
    )
    prepared = prepare_history(history, max_length=1200)
    body = prepared[len(HISTORY_TRUNCATION_NOTICE):]

    assert body.startswith("User: ")


def test_one_exchange_larger_than_the_whole_budget_is_still_bounded():
    """No exchange boundary to cut at — keep the tail rather than nothing."""
    history = "User: q\nAssistant: " + ("filler " * 2000)
    prepared = prepare_history(history, max_length=500)

    assert len(prepared) <= 500 + len(HISTORY_TRUNCATION_NOTICE)
    assert prepared.endswith("filler ")


def test_history_budget_of_zero_drops_history():
    assert prepare_history("User: hi\nAssistant: hello", max_length=0) == ""


def test_empty_history_stays_empty():
    assert prepare_history("") == ""


def test_long_history_cannot_crowd_out_code_context():
    """The two budgets are separate; that is the whole point of splitting them."""
    context = _context(2)
    history = "User: old\nAssistant: " + ("filler " * 5000)

    content = prepare_user_content("what does handler_1 do?", context, history)

    assert "handler_1" in content
    assert "Result 1:" in content
    assert "Result 2:" in content


# ---------------------------------------------------------------------------
# Wiring: the budgets have to apply on the real path
# ---------------------------------------------------------------------------
def test_build_messages_applies_the_context_budget():
    oversized = _context(400)
    messages = build_messages("q", oversized, "")
    user_content = messages[1]["content"]

    assert len(user_content) < len(oversized)
    assert TRUNCATION_NOTICE in user_content


def test_build_messages_applies_the_history_budget():
    history = "User: old\nAssistant: " + ("filler " * 5000)
    messages = build_messages("q", _context(1), history)
    user_content = messages[1]["content"]

    assert len(user_content) < len(history)


def test_build_messages_shape_is_unchanged():
    messages = build_messages("q", _context(1), "User: hi")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "DO NOT guess" in messages[0]["content"]


# ---------------------------------------------------------------------------
# The preview endpoint must report what it dropped
# ---------------------------------------------------------------------------
def test_preview_reports_truncation():
    preview = generate_prompt_preview("q", _context(400), "")
    budget = preview["budget"]

    assert budget["context_truncated"] is True
    assert budget["original_context_chars"] > budget["context_chars"]
    assert budget["max_context_chars"] == MAX_PROMPT_CONTEXT_CHARS


def test_preview_reports_no_truncation_when_none_happened():
    preview = generate_prompt_preview("q", _context(2), "User: hi")
    budget = preview["budget"]

    assert budget["context_truncated"] is False
    assert budget["history_truncated"] is False


def test_preview_context_matches_what_was_sent():
    """A preview showing an untruncated prompt would be worse than none."""
    context = _context(400)
    preview = generate_prompt_preview("q", context, "")

    assert preview["retrieved_context"] in preview["final_prompt_messages"][1]["content"]


def test_preview_keeps_its_existing_keys():
    preview = generate_prompt_preview("q", "ctx", "hist")

    for key in (
        "user_query",
        "retrieved_context",
        "conversation_history",
        "system_prompt",
        "final_prompt_messages",
    ):
        assert key in preview


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0, 500, -1])
def test_below_minimum_context_budget_is_rejected(value, monkeypatch):
    """validate_config() has to catch a budget too small to hold one chunk."""
    import config

    monkeypatch.setattr(config, "MAX_PROMPT_CONTEXT_CHARS", value)
    with pytest.raises(ValueError) as excinfo:
        config.validate_config()

    assert "MAX_PROMPT_CONTEXT_CHARS" in str(excinfo.value)


def test_negative_history_budget_is_rejected(monkeypatch):
    import config

    monkeypatch.setattr(config, "MAX_PROMPT_HISTORY_CHARS", -1)
    with pytest.raises(ValueError) as excinfo:
        config.validate_config()

    assert "MAX_PROMPT_HISTORY_CHARS" in str(excinfo.value)


def test_valid_budgets_pass_validation():
    import config

    config.validate_config()
