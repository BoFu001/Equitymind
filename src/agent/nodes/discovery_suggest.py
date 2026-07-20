"""
src/agent/nodes/discovery_suggest.py

Node: Discovery Suggest
"""

import json
import random

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL
from langgraph.config import get_stream_writer
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint, rprint

client = OpenAI(api_key=OPENAI_API_KEY)


def discovery_suggest(state: AgentState) -> dict:
    """
    Single responsibility: LLM suggests 5 candidate tickers based on user criteria.
    Writes candidate tickers to state["tickers"] so retrieve_sec, stock_snapshots, news can process them.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "discovery", "message": NODE_PROGRESS["discovery_suggest"]})

    # use synthesized question if clarification ran, else the raw user question
    question = state.get("enriched_query") or state["question"]

    rprint(f"  [discovery_suggest] question: {question}")

    ticker_prompt = f"""You are a financial analyst.
The user wants investment recommendations based on their criteria.

USER QUESTION: {question}

Return exactly 5 stock tickers that could match the user's criteria.
IMPORTANT: Only suggest US-listed companies that file 10-K annual reports with the SEC.
Do NOT suggest foreign companies or ADRs (e.g. Alibaba, ASML, Toyota, TSM).
Avoid always suggesting the same popular mega-cap companies (AAPL, MSFT, GOOGL, AMZN, NVDA) unless the user specifically asks for them.
Be creative and consider less obvious but relevant companies that genuinely match the user's criteria.
Exploration seed: {random.randint(1000, 9999)}

Reply with ONLY valid JSON. No markdown, no code fences, no explanation. Example:
{{"tickers": ["JNJ", "WMT", "BRK-B", "PFE", "JPM"]}}"""

    # TODO: Once 20-F pipeline is built, remove the 10-K constraint above.

    ticker_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": ticker_prompt}],
        temperature=0.3,
    )

    try:
        ticker_data = json.loads(ticker_response.choices[0].message.content.strip())
        candidate_tickers = ticker_data.get("tickers", [])
    except Exception:
        gprint(f"  [discovery_suggest] Could not parse candidate tickers")
        candidate_tickers = []

    gprint(f"  [discovery_suggest] Candidates: {candidate_tickers}")
    return {"tickers": candidate_tickers}
