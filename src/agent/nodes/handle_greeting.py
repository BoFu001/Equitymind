"""
src/agent/nodes/handle_greeting.py

Node: Handle Greeting
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL_LIGHT, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters.conversation_formatter import format_conversation_context
from src.agent.capabilities import CAPABILITIES
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

WHAT YOU CAN BE ASKED:
{CAPABILITIES}

CONVERSATION HISTORY:
{conversation_context}

USER MESSAGE: {question}

If this is the first message (no history) — greet them and show what they
can ask, drawing on the questions above rather than naming features. The
example questions are the useful part: someone reading the reply should be
able to copy one and get a real answer back. Naming a capability ("news
sentiment analysis") leaves them still guessing at the wording.
If the user is saying thank you, well done, or giving positive feedback — respond naturally and briefly, then invite them to ask another question.
If the user is saying goodbye — respond warmly and briefly.

Keep the response concise and contextual. Use markdown and emojis where appropriate."""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        # Low but not zero. These replies teach the user what the
        # system does, and at 0.7 the list came back shorter or longer
        # each time — two people asking the same thing learned different
        # things about it. The wording still has to follow whatever
        # language and tone the question arrived in, so not 0 either.
        temperature=0.3,
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
