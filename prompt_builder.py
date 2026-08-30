"""
prompt_builder.py — Modular prompt construction pipeline for DevWhisper.

Separates prompt construction into clear, decoupled stages:
  1. Context preparation: cleaning, formatting, and truncating code context.
  2. Prompt assembly: combining system prompts, user queries, code context, and history into structured message payloads.
"""

import re
from typing import List, Dict, Any
from context_packer import pack_context, estimate_token_count

from config import MAX_PROMPT_CONTEXT_CHARS, MAX_PROMPT_HISTORY_CHARS
from logger import logger

# retrieve() formats each fused chunk as a block beginning "Result <n>:".
# Truncation cuts between those blocks rather than inside one — see
# prepare_context() for why that matters.
_RESULT_BOUNDARY_RE = re.compile(r"(?m)^Result \d+:$")

# Wording kept verbatim from the original implementation: it is the marker
# the model and the existing tests both key off.
TRUNCATION_NOTICE = "\n...[context truncated]"
HISTORY_TRUNCATION_NOTICE = "...[earlier turns omitted]\n"

# Strict system prompt constraining LLM answers to provided codebase context
SYSTEM_PROMPT = """
You are DevWhisper, a strict codebase analysis assistant.

STRICT RULES:
• ONLY use the provided code context
• DO NOT use general knowledge
• DO NOT explain tools or querying
• DO NOT guess
• DO NOT use phrases like "it appears", "it seems", "looks like"

IF ASKED ABOUT FUNCTIONS:
• Extract actual function names from the code
• Respond ONLY in this format:

Functions found:
- In .py: func1, func2

• If multiple files, list each file separately
• If no functions found, say:
"I could not find this in your codebase."

IF ASKED ANYTHING ELSE:
• Answer ONLY if clearly present in code
• Otherwise say:
"I could not find this in your codebase."

STYLE:
• Be direct
• No extra explanation
• Short and voice-friendly
"""

# Shared user instructions appended to every query
USER_INSTRUCTIONS = """
INSTRUCTIONS:
- Answer strictly from code
- Do NOT add explanation unless asked
- Keep output clean and structured
"""


def prepare_context(
    context: str,
    max_length: int | None = None,
    max_tokens: int | None = None,
    enable_compression: bool = True,
) -> str:
    """
    Stage 1: Context Preparation & Token Budgeting.
    Clean, format, compress, and truncate retrieved code context within token limits.
    """
    if not context:
        return ""

    if max_tokens is not None:
        packed, _ = pack_context(context, max_tokens=max_tokens, enable_compression=enable_compression)
        return packed

    cleaned_context = context.strip()

    if max_length is None:
        max_length = MAX_PROMPT_CONTEXT_CHARS
    if not max_length or len(cleaned_context) <= max_length:
        return cleaned_context

    truncated = _truncate_at_result_boundary(cleaned_context, max_length)
    logger.warning(
        "Prompt context truncated from %d to %d characters (budget %d). "
        "Consider lowering RETRIEVAL_TOP_K or INDEX_CHUNK_SIZE for this "
        "repository.",
        len(cleaned_context),
        len(truncated),
        max_length,
    )
    return truncated + TRUNCATION_NOTICE


def _truncate_at_result_boundary(context: str, max_length: int) -> str:
    """
    Trim *context* to *max_length* characters, cutting between result blocks.

    ``context[:max_length]`` lands mid-token: the model is handed half an
    identifier, an unterminated string, and a dangling ``Code:`` header with
    nothing under it. That is precisely the input that makes it start filling
    in the gaps — directly against the "DO NOT guess" rule in
    :data:`SYSTEM_PROMPT`. Dropping whole results instead means everything the
    model does see is complete and true.

    Args:
        context: Cleaned context, expected to be ``Result n:`` blocks.
        max_length: Character budget for the returned string.

    Returns:
        The largest prefix of whole result blocks that fits. If even the first
        block is too large — one enormous chunk — falls back to a hard cut at
        a line boundary, since returning nothing would be worse.
    """
    boundaries = [match.start() for match in _RESULT_BOUNDARY_RE.finditer(context)]

    # Keep the last boundary that still leaves us inside the budget.
    kept = 0
    for start in boundaries:
        if start == 0:
            continue
        if start > max_length:
            break
        kept = start

    if kept:
        return context[:kept].rstrip()

    # A single result exceeds the whole budget. Cut at the last newline so at
    # least the final line the model sees is a complete one.
    hard_cut = context[:max_length]
    newline = hard_cut.rfind("\n")
    if newline > max_length // 2:
        hard_cut = hard_cut[:newline]
    return hard_cut.rstrip()


def prepare_history(history: str, max_length: int | None = None) -> str:
    """
    Trim conversation history to its own budget, keeping the most recent turns.

    History is budgeted separately from code context and given far less room.
    It holds up to ``max_history_per_session`` complete previous answers, so
    under a shared budget one long earlier answer could crowd out the code the
    current question is about — and history is the less valuable of the two.

    Trimming keeps the *end* of the history, because the most recent exchange
    is the one the current question most likely refers back to.

    Args:
        history: The formatted history string from ``SessionManager.get()``.
        max_length: Character budget. None uses
            :data:`MAX_PROMPT_HISTORY_CHARS`; 0 drops history entirely.

    Returns:
        The trimmed history.
    """
    if not history:
        return ""

    if max_length is None:
        max_length = MAX_PROMPT_HISTORY_CHARS
    if max_length == 0:
        return ""
    if len(history) <= max_length:
        return history

    tail = history[-max_length:]
    # Start at an exchange boundary so the prompt never opens mid-sentence.
    boundary = tail.find("User: ")
    if boundary > 0:
        tail = tail[boundary:]
    return HISTORY_TRUNCATION_NOTICE + tail


def prepare_user_content(user_query: str, context: str, history: str = "") -> str:
    """
    Stage 2: User Content Formatting.
    Formats the user question, retrieved code context, and conversation history.

    Both parts are trimmed to their configured budgets here, which is the one
    place every prompt passes through.
    """
    prepared_ctx = prepare_context(context)
    prepared_history = prepare_history(history)
    return f"""
User question:
{user_query}

Code context:
{prepared_ctx}

Conversation history:
{prepared_history}

{USER_INSTRUCTIONS}
"""


def build_messages(user_query: str, context: str, history: str = "") -> List[Dict[str, str]]:
    """
    Stage 3: Complete Prompt Assembly.
    Assembles system and user messages into the OpenAI/Groq chat completions payload format.
    """
    user_content = prepare_user_content(user_query, context, history)
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def generate_prompt_preview(user_query: str, context: str, history: str = "") -> Dict[str, Any]:
    """
    Generate prompt preview displaying retrieved context, system prompt, and final assembled messages.

    The preview reports truncation explicitly. Without that it would show a
    prompt that differs from the one actually sent, which defeats the point of
    a preview endpoint.
    """
    packed_ctx, telemetry = pack_context(context, max_tokens=4096, enable_compression=True)
    messages = build_messages(user_query, packed_ctx, history)
    return {
        "user_query": user_query,
        "retrieved_context": packed_ctx,
        "context_telemetry": telemetry,
        "conversation_history": history,
        "system_prompt": SYSTEM_PROMPT.strip(),
        "final_prompt_messages": messages,
        "budget": {
            "max_context_chars": MAX_PROMPT_CONTEXT_CHARS,
            "max_history_chars": MAX_PROMPT_HISTORY_CHARS,
            "context_chars": len(prepared_ctx),
            "original_context_chars": original_context_chars,
            "context_truncated": len(prepared_ctx) != original_context_chars,
            "history_chars": len(prepared_history),
            "original_history_chars": original_history_chars,
            "history_truncated": len(prepared_history) != original_history_chars,
        },
    }

