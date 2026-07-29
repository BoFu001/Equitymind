"""
src/agent/nodes/determine_data_scope.py

Node: Determine Data Scope

Decides which data sources/signals a question actually needs, so
fetch_all_data / quant_engine / generate_report can skip unneeded
work instead of unconditionally fetching, computing, and formatting
all six signals for every request.

A single, one-shot classification (same pattern as extract_parameters
and classify_sub_intent), NOT an agent loop — the answer is fully
determined by the question's text alone and doesn't depend on any
intermediate result, so an iterative/multi-step decision process adds
cost and latency without adding accuracy. An earlier attempt at
LLM-driven data-source selection (an agent loop deciding which tools
to call before fetch_all_data) was abandoned because quant_engine's
signal engines shared a single market_data dict at the time — skipping
it for one signal silently skipped ALL signals downstream, including
ones that didn't actually need it. That coupling no longer exists
(each signal now has its own independent reader — see src/readers/),
which is why this is being reattempted as a single-shot classifier
rather than a loop.

Runs only on the SPECIFIC_STOCK / COMPARISON path, after tickers are
confirmed present (see graph.py's route_after_extract). The DISCOVERY
path does not call this node — discovery_suggest must already
identify which signals a discovery question needs (e.g. "quality
stocks with low valuation" implies Quality + Valuation) in order to
build its SQL filter, so that node produces signals_needed itself
rather than duplicating the same classification here.

snapshot (company name, price, market cap, etc.) is NOT part of
signals_needed — it is always fetched regardless, since almost any
question may reference basic company info in its answer.
"""

import json

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL_LIGHT
from langgraph.config import get_stream_writer
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)

# The 7 signals/data sources this node can select from (snapshot is
# excluded — see module docstring). Kept as a module-level constant so
# the prompt and the post-response validation step (which drops any
# value the model returns that isn't in this list) stay in sync.
VALID_SIGNALS = [
    "valuation",
    "momentum",
    "risk",
    "quality",
    "consensus",
    "news",
    "financial_history",
]


def determine_data_scope(state: AgentState) -> dict:
    """
    Classifies which of the 7 optional data sources/signals this
    question needs. Returns {"signals_needed": [...]}.

    Defaults to ALL 7 signals if the LLM call fails or returns
    unparseable output — a full-analysis fallback is the safe
    degradation here (matches current unconditional-fetch behavior),
    not an empty list, since silently answering with less data than
    available is worse than doing slightly more work than necessary.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "determine_data_scope", "message": NODE_PROGRESS["determine_data_scope"]})

    question = state.get("contextualized_question") or state["question"]

    prompt = f"""You are a financial data-scope classifier. Decide which
data sources are needed to answer the user's question about a stock.

Available data sources:
- valuation: P/E, P/B ratios and how they compare to peers — for questions about whether a stock is over/undervalued, expensive, or cheap.
- momentum: 12-month price return and 52-week high/low position — for questions about price trend, momentum, or recent performance.
- risk: Beta, Sharpe Ratio, Value at Risk, Max Drawdown — for questions about volatility, risk, or downside.
- quality: Piotroski F-Score financial health checks — for questions about financial health, profitability trends, or fundamentals quality.
- consensus: analyst recommendations, price targets, rating trend — for questions about what analysts think, price targets, or buy/sell ratings.
- news: recent news sentiment — for questions about recent news, headlines, or media coverage.
- financial_history: multi-year revenue/income/balance-sheet trend data — for questions asking how a specific financial figure has changed over multiple years or a specific past fiscal year/quarter.

Rules:
- Select ONLY the data sources the question actually needs to be answered well.
- If the question asks for a full/complete/comprehensive analysis, or doesn't specify a narrow focus, select ALL 7.
- If the question is narrow (e.g. only about analyst opinions, or only about recent news), select only the relevant one(s).
- Company snapshot data (price, market cap, sector) is always included separately and should NOT be listed here.

User question: {question}

Reply with ONLY valid JSON, a single key "signals_needed" with a list of the applicable values from: {VALID_SIGNALS}. No markdown, no explanation. Examples:
{{"signals_needed": ["valuation", "momentum", "risk", "quality", "consensus", "news", "financial_history"]}}
{{"signals_needed": ["consensus"]}}
{{"signals_needed": ["news"]}}
{{"signals_needed": ["valuation", "quality"]}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if not content:
        gprint(f"  [determine_data_scope] Empty response from {LLM_MODEL_LIGHT}, defaulting to all signals")
        return {"signals_needed": list(VALID_SIGNALS)}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        gprint(f"  [determine_data_scope] Invalid JSON: {content}, defaulting to all signals")
        return {"signals_needed": list(VALID_SIGNALS)}

    signals_needed = data.get("signals_needed", [])
    # Drop anything the model returned that isn't a recognized value,
    # rather than trusting it blindly — downstream fetch_all_data
    # dispatches on these strings directly.
    signals_needed = [s for s in signals_needed if s in VALID_SIGNALS]

    if not signals_needed:
        gprint(f"  [determine_data_scope] No valid signals parsed, defaulting to all signals")
        return {"signals_needed": list(VALID_SIGNALS)}

    gprint(f"  [determine_data_scope] Signals needed: {signals_needed}")
    return {"signals_needed": signals_needed}
