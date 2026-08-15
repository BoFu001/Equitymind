"""
src/agent/nodes/discovery_experiment.py

Node: Discovery Experiment
"""

import numpy as np
import psycopg2
from openai import OpenAI
from pydantic import BaseModel

from config import OPENAI_API_KEY, LLM_MODEL, DATABASE_URL
from src.agent.state import AgentState
from src.sec_pipeline.embedder import EMBEDDING_MODEL
from colors import gprint, rprint

client = OpenAI(api_key=OPENAI_API_KEY)


from typing import Literal

QUANT_SIGNALS_FIELDS = [
    # quant_signals — real, ranked columns only (not _data/_computed_at/_period_end)
    "valuation_score",
    "momentum_12_1_score", "position_52w_score",
    "risk_beta_score", "risk_sharpe_score", "risk_var_score", "risk_drawdown_score",
    "quality_score",
    "consensus_recommendation_score", "consensus_upside_score", "consensus_trend_score",
    "short_interest_pct", "days_to_cover",
]

STOCK_UNIVERSE_FIELDS = [
    # stock_universe — single-row-per-ticker current-state table.
    # market_cap lives here, not in financial_history: market cap is a
    # market-data figure (price x shares outstanding), not something
    # any of the three financial statements report, and it is a
    # point-in-time "now" figure rather than a historical multi-period
    # one, so it doesn't fit financial_history's per-quarter row model.
    "market_cap",
]

FINANCIAL_HISTORY_FIELDS = [
    # financial_history — every numeric column (not ticker/period_end/period_type/updated_at)
    "total_revenue", "cost_of_revenue", "gross_profit",
    "research_and_development", "selling_general_and_administration",
    "operating_expense", "operating_income", "ebit", "ebitda", "pretax_income",
    "net_income", "diluted_eps", "basic_eps", "interest_expense",
    "operating_cash_flow", "capital_expenditure", "free_cash_flow",
    "repurchase_of_capital_stock", "cash_dividends_paid",
    "depreciation_amortization_depletion",
    "total_assets", "total_liabilities", "stockholders_equity",
    "cash_and_equivalents", "long_term_debt",
    "current_assets", "current_liabilities",
    "shares_outstanding", "retained_earnings", "net_ppe",
    "accounts_receivable", "inventory", "total_debt",
]

VALID_FIELD_NAMES = Literal[tuple(QUANT_SIGNALS_FIELDS + STOCK_UNIVERSE_FIELDS + FINANCIAL_HISTORY_FIELDS)]


class RankField(BaseModel):
    """One field the user wants to rank by. name must be one of the
    fields this system actually has data for — if the user asks about
    something we don't track (e.g. "beautiful", "tall"), it should not
    appear here at all, not be guessed at."""
    name: VALID_FIELD_NAMES
    order: str          # "ascending" or "descending"
    count: int | None    # this field's own LIMIT, if the sentence gave one
    priority: int         # same number across fields = averaged together;
                           # different numbers = executed in ascending order


class DiscoveryQuery(BaseModel):
    industry: str | None       # user's own wording, or None if no industry mentioned
    fields: list[RankField]    # zero or more ranking fields
    final_count: int | None    # how many results to show at the very end,
                                # distinct from any single field's own count


def extract_discovery_query(question: str) -> DiscoveryQuery:
    """
    Turn a user's free-text question into a structured DiscoveryQuery.
    LLM only transcribes what the sentence says — no ranking logic, no
    judgment. If parsing fails, returns an empty query rather than raising.
    """
    
    extraction_prompt = f'''You are parsing a user\'s stock question into structured
data. Do NOT answer the question or judge anything — only faithfully transcribe
what the sentence says. All logic (ranking, filtering) happens in code afterward.

USER QUESTION: {question}

Extract two things:

1. industry: the user\'s own wording for any industry/theme mentioned (e.g.
   "semiconductor stocks", "robotics companies", "healthcare"). Copy their
   phrasing verbatim, do not translate or normalize it. If no industry is
   mentioned, use null.

2. fields: a list of ranking fields the user mentioned. Each field has:
   - name: the field being ranked. MUST be exactly one of these values —
     do not invent a name, do not use spaces (use underscores), and if the
     user's question does not clearly map to any of these, leave it out of
     the fields list entirely rather than guessing:
     valuation_score, momentum_12_1_score, position_52w_score,
     risk_beta_score, risk_sharpe_score, risk_var_score, risk_drawdown_score,
     quality_score, consensus_recommendation_score, consensus_upside_score,
     consensus_trend_score, short_interest_pct, days_to_cover, market_cap,
     total_revenue, cost_of_revenue, gross_profit, research_and_development,
     selling_general_and_administration, operating_expense, operating_income,
     ebit, ebitda, pretax_income, net_income, diluted_eps, basic_eps,
     interest_expense, operating_cash_flow, capital_expenditure,
     free_cash_flow, repurchase_of_capital_stock, cash_dividends_paid,
     depreciation_amortization_depletion, total_assets, total_liabilities,
     stockholders_equity, cash_and_equivalents, long_term_debt,
     current_assets, current_liabilities, shares_outstanding,
     retained_earnings, net_ppe, accounts_receivable, inventory, total_debt

     When the user says something generic, map it to the closest specific
     field: "risk" alone -> risk_beta_score; "momentum" alone ->
     momentum_12_1_score; "consensus"/"analyst rating" alone ->
     consensus_recommendation_score.
   - order: "ascending" (lowest/cheapest/safest first) or "descending"
     (highest/most/largest first) — watch for negation, e.g. "least favored
     by analysts" is ascending on consensus, not descending
   - count: a number if THIS SPECIFIC field has one attached in the sentence
     (e.g. "the 10 largest" -> count=10 on the market-cap field). Otherwise null.
   - priority: an integer. Assign this by sentence structure, not by which
     fields are involved:
     - If two or more fields are joined by "and"/"but", or simply listed
       together with no containment wording ("highest net income and lowest
       valuation", "low-risk and high-momentum stocks") -> give them the
       SAME priority number. They will be averaged together with equal
       weight, no field is ever treated as more important than another by
       default.
     - If the sentence has a containment structure ("of the 10 largest
       companies, which has the lowest X" / "among X stocks, Y" / "X 里 Y")
       -> give the outer (containing) field a SMALLER priority number than
       the inner (contained) field. Lower number = executed first.
     - A sentence can have more than two priority levels if it nests more
       than once (e.g. "of the 20 largest, the 10 cheapest, the most
       analyst-favored" -> three distinct priority numbers, 1/2/3).

3. final_count: a number if the user asked for a specific count of final
   results that does NOT belong to any single field (e.g. "give me 10
   stocks with low valuation and high quality" -> the 10 applies to the
   combined result, not to valuation or quality individually -> final_count=10,
   and neither field gets its own count). Null if unspecified. Do not
   confuse this with a field-specific count in a nested sentence (e.g. "of
   the 10 largest" -> that 10 belongs to the market-cap field, not here).

EXAMPLES:

"Give me 10 stocks with low valuation and high quality"
{{"industry": null, "fields": [
  {{"name": "valuation_score", "order": "ascending", "count": null, "priority": 1}},
  {{"name": "quality_score", "order": "descending", "count": null, "priority": 1}}
], "final_count": 10}}

"Of the 10 largest companies, which has the lowest valuation?"
{{"industry": null, "fields": [
  {{"name": "market_cap", "order": "descending", "count": 10, "priority": 1}},
  {{"name": "valuation_score", "order": "ascending", "count": null, "priority": 2}}
], "final_count": null}}

"Of the 20 largest companies, the 10 with the lowest valuation, which 5 do
analysts favor most?"
{{"industry": null, "fields": [
  {{"name": "market_cap", "order": "descending", "count": 20, "priority": 1}},
  {{"name": "valuation_score", "order": "ascending", "count": 10, "priority": 2}},
  {{"name": "consensus_recommendation_score", "order": "descending", "count": 5, "priority": 3}}
], "final_count": null}}

"Recommend some low-risk, high-momentum semiconductor stocks"
{{"industry": "semiconductor", "fields": [
  {{"name": "risk_beta_score", "order": "ascending", "count": null, "priority": 1}},
  {{"name": "momentum_12_1_score", "order": "descending", "count": null, "priority": 1}}
], "final_count": null}}

Reply with ONLY valid JSON matching this shape, no markdown, no commentary:
{{"industry": "...", "fields": [{{"name": "...", "order": "...", "count": null, "priority": 1}}], "final_count": null}}'''

    extraction_response = client.responses.parse(
        model=LLM_MODEL,
        input=[{"role": "user", "content": extraction_prompt}],
        temperature=0,
        text_format=DiscoveryQuery,
    )
    try:
        query = extraction_response.output_parsed
    except Exception as e:
        gprint(f"  [extract_discovery_query] Could not parse query: {e}")
        query = DiscoveryQuery(industry=None, fields=[], final_count=None)
    print(f"  [extract_discovery_query] parsed:\n\n{query.model_dump_json(indent=2)}")
    return query



def get_all_tickers() -> list[str]:
    """No industry mentioned — every ticker is in scope."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM stock_universe")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    tickers = [ticker for (ticker,) in rows]
    print(f"  [get_all_tickers] no industry specified, using all {len(tickers)} tickers")
    return tickers


def get_industry_tickers(industry_words: str) -> list[str]:
    """
    RAG retrieval: embed the industry words, compare against every
    company's overview embedding, print scores for debugging, return
    only the ticker list — the caller doesn't need to know about scores.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, overview_embedding FROM stock_universe WHERE overview_embedding IS NOT NULL"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    query_vector = np.array(
        client.embeddings.create(model=EMBEDDING_MODEL, input=industry_words).data[0].embedding
    )

    scored = []
    for ticker, embedding in rows:
        if isinstance(embedding, str):
            company_vector = np.array([float(x) for x in embedding.strip("[]").split(",")])
        else:
            company_vector = np.array(embedding, dtype=float)
        similarity = float(
            np.dot(query_vector, company_vector)
            / (np.linalg.norm(query_vector) * np.linalg.norm(company_vector))
        )
        scored.append((ticker, similarity))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    print(f"  [get_industry_tickers] scores for {industry_words!r} ({len(scored)} tickers):")
    for ticker, score in scored[:20]:
        print(f"    {ticker:6s} {score:.4f}")

    return [ticker for ticker, score in scored]



def get_field_source(field_name: str) -> str:
    """
    Which table does this field live in? quant_signals holds the six
    normalized signal scores; financial_history holds the raw dollar
    figures from the three financial statements; stock_universe holds
    market_cap (see STOCK_UNIVERSE_FIELDS above for why). Pure lookup,
    no database call — used before querying to decide which table to
    hit.
    """
    if field_name in QUANT_SIGNALS_FIELDS:
        return "quant_signals"
    elif field_name in FINANCIAL_HISTORY_FIELDS:
        return "financial_history"
    else:
        return "stock_universe"


def fetch_field_values(tickers: list[str], field_name: str) -> dict[str, float]:
    """
    Given a candidate pool and a field name, returns {ticker: real_value}
    for every ticker that has a usable value — no percentile math here,
    just the raw lookup.
    quant_signals fields: current value, straightforward lookup, one row
    per ticker, no period concept.
    financial_history fields: each ticker has many historical rows (one
    per quarter). We deliberately use the SECOND-most-recent quarter
    (rn=2), not the latest — the latest quarter is often still being
    backfilled right after earnings (income-statement fields arrive
    first, balance-sheet fields lag), so using it would systematically
    exclude companies that just reported, which is exactly the opposite
    of what a "which stock is cheapest right now" query needs. The
    second-most-recent quarter trades a bit of recency for having a
    complete, comparable snapshot across the whole candidate pool.
    Verified against 5 tickers spanning standard and non-standard
    fiscal years (AAPL, MSFT, JPM, ORCL, COST) — all fully populated
    at rn=2.
    stock_universe fields (market_cap): current value, straightforward
    lookup, one row per ticker, no period concept — same shape as
    quant_signals, just a different table.
    Tickers whose value is NULL at that period (or who have no matching
    row at all) are simply absent from the returned dict — the caller
    treats "not in this dict" as "excluded from this ranking," not as
    an error.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    source = get_field_source(field_name)
    if source == "quant_signals":
        query = f"""
            SELECT ticker, {field_name}
            FROM quant_signals
            WHERE ticker = ANY(%s) AND {field_name} IS NOT NULL
        """
        cursor.execute(query, (tickers,))
    elif source == "financial_history":
        query = f"""
            SELECT ticker, {field_name}
            FROM (
                SELECT ticker, {field_name},
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY period_end DESC) AS rn
                FROM financial_history
                WHERE ticker = ANY(%s) AND period_type = 'quarterly'
            ) ranked
            WHERE rn = 2 AND {field_name} IS NOT NULL
        """
        cursor.execute(query, (tickers,))
    else:
        query = f"""
            SELECT ticker, {field_name}
            FROM stock_universe
            WHERE ticker = ANY(%s) AND {field_name} IS NOT NULL
        """
        cursor.execute(query, (tickers,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    result = {}
    for ticker, value in rows:
        result[ticker] = float(value)

    print(f"  [fetch_field_values] {field_name}: {len(result)} values")
    return result


def compute_percentiles(values: dict[str, float], order: str) -> dict[str, float]:
    """
    Given {ticker: real_value} and a direction, returns {ticker: percentile}
    where the best-ranked ticker gets 1.0 and the worst gets 0.0, evenly
    spaced in between. "Best" means smallest value if order="ascending"
    (e.g. cheapest valuation), largest value if order="descending" (e.g.
    highest net income).

    A single ticker returns percentile 1.0 for itself (no meaningful
    ranking possible with n=1, but this avoids a divide-by-zero on
    (n-1)).
    """
    if not values:
        return {}

    reverse = (order == "descending")
    ranked_tickers = sorted(values.keys(), key=lambda t: values[t], reverse=reverse)

    n = len(ranked_tickers)
    if n == 1:
        return {ranked_tickers[0]: 1.0}

    return {
        ticker: (n - 1 - rank) / (n - 1)
        for rank, ticker in enumerate(ranked_tickers)
    }

def filter_complete_candidates(tickers: list[str], fields: list[RankField]) -> list[str]:
    """
    Before any ranking happens, keep only tickers that have a real value
    on EVERY field this query will use — like requiring both a language
    score and a math score before a student can be considered at all,
    not computing an average from whichever score happens to exist. A
    ticker missing any one required field is not a partial candidate;
    it has no data at all for this query.

    Implemented as a set intersection across all fields' "has a value"
    sets — a ticker survives only if it appears in every field's set.
    """
    if not fields:
        return tickers

    eligible = set(tickers)
    for field in fields:
        values = fetch_field_values(tickers, field.name)
        eligible &= set(values.keys())

    return [t for t in tickers if t in eligible]

def group_fields_by_priority(fields: list[RankField]) -> list[list[RankField]]:
    """
    Pure data reshaping — no database, no LLM. Takes the flat list the
    LLM produced and turns it into an ordered chain of stages: fields
    sharing the same priority number become one stage (averaged
    together later); different priority numbers become separate,
    ordered stages (executed in sequence later).
    Example:
        [market_cap(p=1), risk(p=2), consensus(p=2)]
        -> [[market_cap], [risk, consensus]]
    """
    if not fields:
        return []
    priorities = sorted(set(field.priority for field in fields))
    stages = []
    for priority in priorities:
        stage = [field for field in fields if field.priority == priority]
        stages.append(stage)
    print(f"  [group_fields_by_priority] {len(stages)} stage(s):")
    for i, stage in enumerate(stages, 1):
        for f in stage:
            print(f"    Stage {i}: name={f.name}, order={f.order}, count={f.count}, priority={f.priority}")
    return stages


def execute_stage(tickers: list[str], stage: list[RankField]) -> list[str]:
    """
    Runs one stage of the ranking chain: given a candidate pool and the
    field(s) sharing this stage's priority, returns the resulting
    ordered ticker list, truncated by count if any field in the stage
    specified one.

    Single field: fetch real values, sort by order, done — no percentile
    math needed since there's nothing to average against.

    Multiple fields (same priority = averaged together): first drop any
    ticker missing a value on ANY of this stage's fields (filter_complete_candidates),
    then compute each remaining field's percentile within this stage's
    pool, average them per ticker, sort by the average.

    count is read from whichever field in the stage carries one — by
    design (see extract_discovery_query's prompt) only one field per
    stage should ever have a non-null count, since a multi-field
    (averaged) stage has no single field's LIMIT to apply.
    """
    if len(stage) == 1:
        field = stage[0]
        values = fetch_field_values(tickers, field.name)
        ranked = sorted(values.keys(), key=lambda t: values[t], reverse=(field.order == "descending"))
        count = field.count
    else:
        complete_tickers = filter_complete_candidates(tickers, stage)
        all_percentiles = {}
        for field in stage:
            values = fetch_field_values(complete_tickers, field.name)
            percentiles = compute_percentiles(values, field.order)
            for ticker, pct in percentiles.items():
                all_percentiles.setdefault(ticker, []).append(pct)

        averaged = {ticker: sum(pcts) / len(pcts) for ticker, pcts in all_percentiles.items()}
        ranked = sorted(averaged.keys(), key=lambda t: averaged[t], reverse=True)
        count = stage[0].count

    return ranked[:count] if count else ranked


def discovery_experiment(state: AgentState) -> dict:
    from src.agent.nodes.discovery_experiment_count import determine_stage_counts

    question = input("Enter question: ")
    query = extract_discovery_query(question)
    if query.industry is None:
        tickers = get_all_tickers()
    else:
        tickers = get_industry_tickers(query.industry)

    stages = group_fields_by_priority(query.fields)
    stages_with_counts = determine_stage_counts(stages, len(tickers))

    for stage in stages_with_counts:
        tickers = execute_stage(tickers, stage)

    tickers = tickers[:query.final_count] if query.final_count else tickers
    return {"tickers": tickers}


if __name__ == "__main__":
    result = discovery_experiment({})
    print(result["tickers"])