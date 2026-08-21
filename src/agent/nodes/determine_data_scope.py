"""
src/agent/nodes/determine_data_scope.py

Node: Determine Data Scope

Decides which data sources/signals a question actually needs, so
fetch_data / quant_engine / generate_report can skip unneeded
work instead of unconditionally fetching, computing, and formatting
all nine data sources for every request.

A single, one-shot classification (same pattern as extract_tickers
and classify_sub_intent), NOT an agent loop — the answer is fully
determined by the question's text alone and doesn't depend on any
intermediate result, so an iterative/multi-step decision process adds
cost and latency without adding accuracy. An earlier attempt at
LLM-driven data-source selection (an agent loop deciding which tools
to call before fetch_data) was abandoned because quant_engine's
signal engines shared a single market_data dict at the time — skipping
it for one signal silently skipped ALL signals downstream, including
ones that didn't actually need it. That coupling no longer exists
(each signal now has its own independent reader — see src/readers/),
which is why this is being reattempted as a single-shot classifier
rather than a loop.

Both paths reach this node: SPECIFIC_STOCK / COMPARISON after tickers
are extracted, and DISCOVERY after its candidate pool is built. The
Discovery path used to bypass it, which meant data_scope was never set
and fetch_data read the empty list as "fetch everything" — nine sources
for every company in a pool that can run to a dozen or more.

Scope is decided from the question, not from the ranking fields
Discovery parsed out of it, because the two answer different questions.
"Which healthcare companies have major recent risks" parses to
risk_beta_score, a measure of price volatility, while what the question
needs is sec_filing — and no numeric field maps to that at all.

snapshot (company name, price, market cap, etc.) is NOT part of
data_scope — it is always fetched regardless, since almost any
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

# The 9 signals/data sources this node can select from (snapshot is
# excluded — see module docstring). Kept as a module-level constant so
# the prompt and the post-response validation step (which drops any
# value the model returns that isn't in this list) stay in sync.
VALID_DATA_SCOPES = [
    "sec_filing",
    "valuation",
    "momentum",
    "risk",
    "quality",
    "consensus",
    "news",
    "financial_history",
    "short",
]


def determine_data_scope(state: AgentState) -> dict:
    """
    Classifies which of the 9 optional data sources/signals this
    question needs. Returns {"data_scope": [...]}.

    Defaults to ALL 9 if the LLM call fails or returns
    unparseable output — a full-analysis fallback is the safe
    degradation here (matches current unconditional-fetch behavior),
    not an empty list, since silently answering with less data than
    available is worse than doing slightly more work than necessary.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "determine_data_scope", "message": NODE_PROGRESS["determine_data_scope"]})

    # Same precedence as fetch_data. After a multi-turn Discovery
    # exchange the latest message is a fragment ("the cheapest ones"),
    # and only the enriched question still carries what was asked for.
    question = state.get("enriched_query") or state.get("contextualized_question") or state["question"]

    prompt = f"""You are a financial data-scope classifier. Decide which
data sources are needed to answer the user's question about a stock.

Available data sources:
- sec_filing: business overview, risk factors, and management discussion & analysis (MD&A) from the company's 10-K — for questions asking what the company's 10-K says about its business, risks, or management's commentary. Does not cover financial statements, executive compensation, or other 10-K sections.
- valuation: P/E, P/B ratios and how they compare to peers — for questions about whether a stock is over/undervalued, expensive, or cheap.
- momentum: 12-month price return and 52-week high/low position — for questions about price trend, momentum, or recent performance.
- risk: Beta, Sharpe Ratio, Value at Risk, Max Drawdown — for questions about volatility, risk, or downside.
- quality: Piotroski F-Score financial health checks — for questions about financial health, profitability trends, or fundamentals quality.
- consensus: analyst recommendations, price targets, rating trend — for questions about what analysts think, price targets, or buy/sell ratings.
- news: recent news sentiment — for questions about recent news, headlines, or media coverage.
- financial_history: reported figures from the income statement, cash flow statement and balance sheet, by fiscal year and quarter — revenue, profit, EPS, assets, debt, cash flow, capital expenditure, dividends paid, and so on. Use this for any question naming a financial statement line item, whether for one period or across several.
- short: short interest percentage, days to cover, and month-over-month change in shares sold short — for questions about short selling, whether a stock is heavily shorted, or short squeeze risk.

Rules:
- Select ONLY the data sources the question actually needs to be answered well.
- If the question asks for a full/complete/comprehensive analysis, or doesn't specify a narrow focus, select ALL 9.
- If the question is narrow (e.g. only about analyst opinions, or only about recent news), select only the relevant one(s).
- Company snapshot data (price, market cap, sector) is always included separately and should NOT be listed here.

User question: {question}

Reply with ONLY valid JSON, a single key "data_scope" with a list of the applicable values from: {VALID_DATA_SCOPES}. No markdown, no explanation. Examples:
{{"data_scope": ["sec_filing", "valuation", "momentum", "risk", "quality", "consensus", "news", "financial_history", "short"]}}
{{"data_scope": ["consensus"]}}
{{"data_scope": ["news"]}}
{{"data_scope": ["valuation", "quality"]}}
{{"data_scope": ["short"]}}
{{"data_scope": ["sec_filing"]}}
{{"data_scope": ["financial_history"]}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if not content:
        gprint(f"  [determine_data_scope] Empty response from {LLM_MODEL_LIGHT}, defaulting to all signals")
        return {"data_scope": list(VALID_DATA_SCOPES)}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        gprint(f"  [determine_data_scope] Invalid JSON: {content}, defaulting to all signals")
        return {"data_scope": list(VALID_DATA_SCOPES)}

    data_scope = data.get("data_scope", [])
    # Drop anything the model returned that isn't a recognized value,
    # rather than trusting it blindly — downstream fetch_data
    # dispatches on these strings directly.
    data_scope = [s for s in data_scope if s in VALID_DATA_SCOPES]

    if not data_scope:
        gprint(f"  [determine_data_scope] DEBUG raw content from LLM: {content}")
        gprint(f"  [determine_data_scope] No valid signals parsed, defaulting to all signals")
        return {"data_scope": list(VALID_DATA_SCOPES)}

    gprint(f"  [determine_data_scope] Data scope: {data_scope}")
    return {"data_scope": data_scope}
