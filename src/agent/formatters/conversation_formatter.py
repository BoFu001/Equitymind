"""
src/agent/formatters/conversation_formatter.py

Formats prior conversation turns for the LLM's context window.

Split out from the former single-file formatters.py on 2026-07-27,
alongside the per-signal formatters. Not tied to any quant signal or
data source — this formats state["messages"], the conversation
history itself.
"""


def format_conversation_context(messages: list, limit: int, max_chars: int = None) -> str:
    conversation_context = ""
    for msg in messages[-limit:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if max_chars:
            content = content[:max_chars]
        conversation_context += f"{role.upper()}: {content}\n"
    return conversation_context
