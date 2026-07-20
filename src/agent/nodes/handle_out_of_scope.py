"""
src/agent/nodes/handle_out_of_scope.py

Node: Handle Out of Scope
"""

import re
import time

from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint
from config import APP_NAME


def handle_out_of_scope(state: AgentState) -> dict:
    """
    Returns a polite refusal for out-of-scope questions.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "out_of_scope", "message": NODE_PROGRESS["out_of_scope"]})

    answer = f"""I'm {APP_NAME}, an AI investment research assistant. I specialise in stock analysis, company research, and investment insights.

I can help you with:
- 📊 Analysing a specific stock (e.g. "Analyse Apple")
- ⚖️ Comparing companies (e.g. "Compare NVIDIA and Microsoft")
- 🔍 Finding investment opportunities (e.g. "Find low risk stocks")
- 📰 News and sentiment analysis
- ⚠️ Risk analysis from SEC filings

What stock would you like me to research?"""

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)
            time.sleep(0.03)

    gprint(f"  [handle_out_of_scope] Response generated ({len(answer)} chars)")
    return {"answer": answer}
