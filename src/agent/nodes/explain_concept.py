"""
src/agent/nodes/explain_concept.py

Node: Explain Concept (GENERAL_KNOWLEDGE handler)
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def explain_concept(state: AgentState) -> dict:
    """
    Answers general conceptual questions about investing.
    No tools, no SEC/market/news data — pure LLM explanation.
    Ends with a soft invitation back into TASK territory.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "explain_concept", "message": NODE_PROGRESS["explain_concept"]})

    question = state["question"]
    messages = state.get("messages") or []

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

The user is asking a general conceptual question about investing — they want to learn, not get a specific recommendation.
Give a clear, friendly, beginner-appropriate explanation. Keep it concise — a few short paragraphs, not an exhaustive essay.
Do NOT ask about sector preference, risk tolerance, or any other specific criteria — that would be premature for someone at this stage.
End with a brief, natural invitation: if they want help finding or analysing a specific stock once they're ready, they can just ask.

CONVERSATION HISTORY (for context):
{conversation_context if conversation_context else "NONE — this is the first message in this session."}

USER QUESTION: {question}

Use markdown and emojis sparingly where it aids clarity."""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    gprint(f"  [explain_concept] Explanation generated ({len(answer)} chars)")
    return {"answer": answer}
