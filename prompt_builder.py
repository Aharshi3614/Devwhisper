"""
prompt_builder.py — Modular prompt construction pipeline for DevWhisper.

Separates prompt construction into clear, decoupled stages:
  1. Context preparation: cleaning, formatting, and truncating code context.
  2. Prompt assembly: combining system prompts, user queries, code context, and history into structured message payloads.
"""

from typing import List, Dict, Any

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


def prepare_context(context: str, max_length: int | None = None) -> str:
    """
    Stage 1: Context Preparation.
    Clean, format, and optionally truncate retrieved code context before embedding in prompts.
    """
    if not context:
        return ""

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
    prepared_ctx = prepare_context(context)
    messages = build_messages(user_query, context, history)
    return {
        "user_query": user_query,
        "retrieved_context": prepared_ctx,
        "conversation_history": history,
        "system_prompt": SYSTEM_PROMPT.strip(),
        "final_prompt_messages": messages,
    }
