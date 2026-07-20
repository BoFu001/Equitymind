"""
src/agent/nodes/handle_no_ticker.py

Node: Handle No Ticker
"""

import re
import time

from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint


def handle_no_ticker(state: AgentState) -> dict:
    """
    User asked a valid financial question but no ticker could be extracted.
    Different from out_of_scope — the intent was valid, just no company identified.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "no_ticker", "message": NODE_PROGRESS["no_ticker"]})

    sub_intent = state.get("sub_intent", "")

    if sub_intent == "COMPARISON":
        answer = f"""I couldn't identify which companies you want to compare.

Please name the companies specifically, for example:
- "Compare Apple and Microsoft"
- "Compare AAPL vs GOOGL"
- "Tesla versus BMW"

Note: Foreign companies like Airbus, Toyota, ASML, Alibaba are not yet supported (coming soon)."""

    else:
        answer = f"""I couldn't identify which company or stock you are asking about.

Please name the company specifically, for example:
- "Analyse Apple"
- "What are NVIDIA's risks?"
- "Tell me about Tesla"

Note: Foreign companies like Airbus, Toyota, ASML, Alibaba are not yet supported (coming soon)."""

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)
            time.sleep(0.03)

    gprint(f"  [handle_no_ticker] Response generated ({len(answer)} chars)")
    return {"answer": answer}
