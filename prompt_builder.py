"""
prompt_builder.py — Modular prompt construction pipeline for DevWhisper.

Separates prompt construction into clear, decoupled stages:
  1. Context preparation: cleaning, formatting, and truncating code context.
  2. Prompt assembly: combining system prompts, user queries, code context, and history into structured message payloads.
"""

from typing import List, Dict, Any
from context_packer import pack_context, estimate_token_count

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
    if max_length and len(cleaned_context) > max_length:
        cleaned_context = cleaned_context[:max_length] + "\n...[context truncated]"

    return cleaned_context


def prepare_user_content(user_query: str, context: str, history: str = "") -> str:
    """
    Stage 2: User Content Formatting.
    Formats the user question, retrieved code context, and conversation history.
    """
    prepared_ctx = prepare_context(context)
    return f"""
User question:
{user_query}

Code context:
{prepared_ctx}

Conversation history:
{history}

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
    }

