"""
src/agent/nodes/extract_tickers.py

Node: Extract Parameters
"""

import json

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL_LIGHT, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters.conversation_formatter import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_tickers(state: AgentState) -> dict:
    """
    Extracts ticker(s) from the user's question.
    Returns the list of all tickers mentioned.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "extract", "message": NODE_PROGRESS["extract"]})

    question = state.get("contextualized_question") or state["question"]
    messages = state.get("messages") or []
    session_memory = state.get("session_memory") or {}
    last_tickers = (session_memory.get("structured") or {}).get("last_tickers", [])

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are a financial data extractor.
Extract the stock ticker(s) from the user question.

Rules:
- tickers: list of ALL stock ticker symbols. Convert ANY company name to its ticker symbol. If no company or ticker mentioned, return [].
- If the user refers to a previously discussed company using a pronoun ("this", "it", "that") instead of naming it, include the ticker from LAST TICKERS FROM PREVIOUS TURN below, IN ADDITION TO any newly named company in the current question. Example: if LAST TICKERS = ["GOOGL"] and the user asks "is this better than Tesla?", return ["GOOGL", "TSLA"] — both the referenced company and the newly named one.
- If the user refers to the whole previous list ("last", "those", "them", "the suggested ones"), return all of LAST TICKERS FROM PREVIOUS TURN.
- Examples of conversions: Apple → AAPL, Microsoft → MSFT, Tesla → TSLA, NVIDIA → NVDA, Google → GOOGL, Amazon → AMZN, Alibaba → BABA, Meta → META, Samsung → 005930.KS, Tencent → 0700.HK

CONVERSATION HISTORY (for context):
{conversation_context}
User question: {question}

LAST TICKERS FROM PREVIOUS TURN:
{last_tickers if last_tickers else "None"}

Reply with ONLY valid JSON. No markdown, no code fences, no explanation. Example:
{{"tickers": ["AAPL"]}}
{{"tickers": ["AAPL", "MSFT"]}}
{{"tickers": ["BABA"]}}
{{"tickers": ["AMZN"]}}
{{"tickers": []}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if not content:
        gprint(f"  [extract_tickers] Empty response from {LLM_MODEL_LIGHT}, using defaults")
        return {"tickers": []}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        gprint(f"  [extract_tickers] Invalid JSON: {content}")
        return {"tickers": []}

    tickers = data.get("tickers", [])

    gprint(f"  [extract_tickers] Tickers: {tickers}")
    return {"tickers": tickers}
