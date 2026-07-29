"""
src/agent/nodes/handle_greeting.py

Node: Handle Greeting
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters.conversation_formatter import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def handle_greeting(state: AgentState) -> dict:
    """
    Returns a friendly greeting and explains what this app can do.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "greeting", "message": NODE_PROGRESS["greeting"]})

    messages = state.get("messages") or []
    question = state["question"]

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

CONVERSATION HISTORY:
{conversation_context}

USER MESSAGE: {question}

If this is the first message (no history) — introduce yourself warmly and explain what you can do.
If the user is saying thank you, well done, or giving positive feedback — respond naturally and briefly, then invite them to ask another question.
If the user is saying goodbye — respond warmly and briefly.

Keep the response concise and contextual. Use markdown and emojis where appropriate."""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    gprint(f"  [handle_greeting] Greeting generated ({len(answer)} chars)")

    return {"answer": answer}
