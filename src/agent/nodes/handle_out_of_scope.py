"""
src/agent/nodes/handle_out_of_scope.py

Node: Handle Out of Scope
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL_LIGHT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.capabilities import CAPABILITIES
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def handle_out_of_scope(state: AgentState) -> dict:
    """
    Returns a polite refusal for out-of-scope questions.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "out_of_scope", "message": NODE_PROGRESS["out_of_scope"]})

    question = state["question"]

    # Written by a model rather than returned as a fixed string, for one
    # reason: the fixed string was English. A question asked in Chinese
    # got the refusal and the whole example list back in English, which
    # reads as though the system did not understand — the opposite of
    # what a refusal should convey. The light model is enough here; the
    # examples are supplied, and the work is translating them and
    # matching the user's tone.
    prompt = f"""You are {APP_NAME}, an AI investment research assistant. You work
from company filings, market data, news and quantitative signals.

WHAT YOU CAN BE ASKED:
{CAPABILITIES}

USER MESSAGE: {question}

This question is outside what you cover. Say so in one line, without
apology, then show what can be asked instead.

Keep all five groups above, with their headings and at least one example
question each. This is the only place a user learns what the system can
do, and dropping the screening or financial-statement examples to save
space leaves them not knowing those questions are possible at all. Do
not replace an example with a description of it: "screen the universe"
teaches nothing that "the five lowest-risk stocks" does not teach
better.

Reply in the language the user wrote in, translating the headings and
the example questions if that language is not English. Use markdown and
emojis where appropriate."""

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

    gprint(f"  [handle_out_of_scope] Response generated ({len(answer)} chars)")
    return {"answer": answer}
