"""
src/agent/nodes/discovery_experiment.py

Node: Discovery Experiment
"""

import numpy as np
import psycopg2
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, DATABASE_URL
from src.agent.state import AgentState
from src.agent.discovery_types import (
    QUANT_SIGNALS_FIELDS,
    FINANCIAL_HISTORY_FIELDS,
    RankField,
    DiscoveryQuery,
)
from src.sec_pipeline.embedder import EMBEDDING_MODEL
from colors import gprint, rprint
from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS

client = OpenAI(api_key=OPENAI_API_KEY)



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


# Tags a company's own overview may phrase differently from another's.
# Each 10-K is summarised independently, so the same concept surfaces
# under variant spellings — "ai" and "artificial intelligence" split
# nine companies each with only SNOW in common, which silently excluded
# NVDA, MSFT and GOOGL from any query resolving to the longer form
# (Research Log 05 3.6). Normalisation lives here rather than in the
# database so that llm_tags stays the raw extraction record: changing a
# rule takes effect on the next query with no data migration, and the
# original wording each filing used is never overwritten.
TAG_SYNONYMS = {
    "ai": "artificial intelligence",
    "property-casualty insurance": "property and casualty insurance",
    "broadband": "broadband services",
}


def _build_tag_index() -> dict[str, list[str]]:
    """Reads every company's raw tags and returns {tag: [tickers]}.

    Rebuilt per call rather than cached: the aggregation is a single
    SELECT over ~250 rows next to an LLM call that costs three orders of
    magnitude more, and Discovery already queries this table elsewhere in
    the same node. Caching would buy nothing and add a staleness mode.

    Plural merging is a rule (strip a trailing "s" only when the singular
    already exists as a tag in its own right, so "logistics" is left
    alone); anything a rule cannot express goes in TAG_SYNONYMS.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT ticker, llm_tags FROM stock_universe WHERE llm_tags IS NOT NULL"
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    vocabulary = {t.strip().lower() for _, tags in rows for t in tags}

    def normalise(tag: str) -> str:
        tag = tag.strip().lower()
        if tag in TAG_SYNONYMS:
            return TAG_SYNONYMS[tag]
        elif tag.endswith("s") and tag[:-1] in vocabulary:
            return tag[:-1]
        else:
            return tag

    index: dict[str, set[str]] = {}
    for ticker, tags in rows:
        for tag in tags:
            index.setdefault(normalise(tag), set()).add(ticker)

    return {tag: sorted(tickers) for tag, tickers in sorted(index.items())}


def get_industry_tickers(industry_words: str) -> list[str]:
    """
    Tag-based retrieval: asks the LLM to pick the single closest tag
    from the deduplicated tag vocabulary (extracted from each
    company's 10-K by update_industry_tags.py, normalised here for
    singular/plural and synonym variants) and looks up the exact
    ticker list for that tag.

    Verified (2026-08-16, Research Log 04) to produce a cleaner
    candidate pool than whole-overview embedding similarity — no
    similarity-threshold guesswork, exact lookup instead of an
    unbounded ranking with no natural cutoff.

    If the LLM finds no reasonable match in the vocabulary, returns an
    empty list rather than falling back to embedding similarity — a
    confident "no match" is preferred over a noisy guess.
    """
    tag_to_tickers = _build_tag_index()
    all_tags = list(tag_to_tickers.keys())

    tag_list_str = ", ".join(all_tags)
    prompt = f"""You are matching a user's industry/theme question to the SINGLE closest tag from a fixed vocabulary.

Available tags (pick exactly one, verbatim, from this list):
{tag_list_str}

User's question: "{industry_words}"

Respond with ONLY the exact tag text from the list above that best matches what the user is asking about. If nothing in the list is a reasonable match, respond with exactly: NONE

Your answer (just the tag, nothing else):"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    if answer == "NONE" or answer not in tag_to_tickers:
        gprint(f"  [get_industry_tickers] {industry_words!r} -> no matching tag found")
        return []

    tickers = tag_to_tickers[answer]
    gprint(f"  [get_industry_tickers] {industry_words!r} -> tag: {answer!r} ({len(tickers)} tickers)")
    return tickers




def get_field_source(field_name: str) -> str:
    """
    Which table does this field live in? quant_signals holds the six
    normalized signal scores; financial_history holds the raw dollar
    figures from the three financial statements; stock_universe holds
    market_cap (see STOCK_UNIVERSE_FIELDS in discovery_types.py for
    why). Pure lookup,
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

    Ties get identical percentiles (fractional ranking): tickers sharing
    the same real value are grouped together and all assigned the AVERAGE
    of the rank positions their group occupies, rather than being split
    across adjacent ranks by an arbitrary tiebreaker (e.g. ticker order).
    Example: values [100, 90, 90, 80] (descending) — the two 90s would
    occupy ranks 1 and 2 (0-indexed) if untied; tied, they both get the
    average rank 1.5, so both receive the same percentile.

    A single ticker returns percentile 1.0 for itself (no meaningful
    ranking possible with n=1, but this avoids a divide-by-zero on
    (n-1)).
    """
    if not values:
        return {}

    n = len(values)
    if n == 1:
        ticker = next(iter(values))
        return {ticker: 1.0}

    if order == "descending":
        sorted_tickers = sorted(values.keys(), key=lambda t: -values[t])
    else:
        sorted_tickers = sorted(values.keys(), key=lambda t: values[t])

    # Group tickers by identical value, in sorted order — each group is
    # a contiguous run of ranks that all share the same real value.
    groups = []
    current_group = [sorted_tickers[0]]
    for ticker in sorted_tickers[1:]:
        if values[ticker] == values[current_group[-1]]:
            current_group.append(ticker)
        else:
            groups.append(current_group)
            current_group = [ticker]
    groups.append(current_group)

    percentiles = {}
    rank = 0
    for group in groups:
        group_size = len(group)
        avg_rank = rank + (group_size - 1) / 2
        percentile = (n - 1 - avg_rank) / (n - 1)
        for ticker in group:
            percentiles[ticker] = percentile
        rank += group_size

    return percentiles

def filter_complete_candidates(tickers: list[str], fields: list[RankField]) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Before any ranking happens, keep only tickers that have a real value
    on EVERY field this query will use — like requiring both a language
    score and a math score before a student can be considered at all,
    not computing an average from whichever score happens to exist. A
    ticker missing any one required field is not a partial candidate;
    it has no data at all for this query.

    Implemented as a set intersection across all fields' "has a value"
    sets — a ticker survives only if it appears in every field's set.

    Also returns field_values — {field_name: {ticker: value}} — the
    data already fetched while checking completeness, spanning the
    ORIGINAL tickers (not yet narrowed down). The caller must subset
    this to complete_tickers before using it for percentile math.
    Safe to reuse now that compute_percentiles uses fractional ranking
    for ties, so subsetting an already-fetched dict and re-querying the
    database both produce identical percentiles regardless of dict
    iteration order.
    """
    if not fields:
        return tickers, {}

    eligible = set(tickers)
    field_values = {}
    for field in fields:
        values = fetch_field_values(tickers, field.name)
        field_values[field.name] = values
        missing = set(tickers) - set(values.keys())
        if missing:
            gprint(f"  [filter_complete_candidates] {field.name}: excluded {sorted(missing)} (missing this field)")
        eligible &= set(values.keys())

    complete_tickers = [t for t in tickers if t in eligible]
    return complete_tickers, field_values

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
    design (see extract_discovery_query in prepare_discovery.py)
    only one field per
    stage should ever have a non-null count, since a multi-field
    (averaged) stage has no single field's LIMIT to apply.
    """
    if len(stage) == 1:
        field = stage[0]
        values = fetch_field_values(tickers, field.name)
        ranked = sorted(values.keys(), key=lambda t: values[t], reverse=(field.order == "descending"))
        count = field.count
        gprint(f"  [execute_stage] single field: {field.name} ({field.order}) | pool: {len(tickers)} -> {len(values)} | count: {count} | top: {ranked[:3]}")
    else:
        complete_tickers, field_values = filter_complete_candidates(tickers, stage)
        all_percentiles = {}
        for field in stage:
            values = {t: field_values[field.name][t] for t in complete_tickers}
            percentiles = compute_percentiles(values, field.order)
            for ticker, pct in percentiles.items():
                all_percentiles.setdefault(ticker, []).append(pct)

        averaged = {ticker: sum(pcts) / len(pcts) for ticker, pcts in all_percentiles.items()}
        ranked = sorted(averaged.keys(), key=lambda t: averaged[t], reverse=True)
        count = stage[0].count
        field_names = [f.name for f in stage]
        gprint(f"  [execute_stage] averaged fields: {field_names} | pool: {len(tickers)} -> {len(complete_tickers)} | count: {count} | top: {ranked[:3]}")

    return ranked[:count] if count else ranked


def discovery_experiment(state: AgentState) -> dict:
    from src.agent.nodes.discovery_experiment_count import determine_stage_counts

    # writer = get_stream_writer()
    # writer({"type": "progress", "node": "discovery", "message": NODE_PROGRESS["discovery_suggest"]})

    # The parse arrives already done. prepare_discovery ran it to decide
    # whether this question was executable at all, so re-running it here
    # would spend a second call to reach a verdict that could differ from
    # the one the gate acted on.
    query = DiscoveryQuery(**state["discovery_query"])

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
    # Standalone harness. In the graph the parse comes from
    # prepare_discovery; here it is produced locally so this node can
    # still be exercised on its own.
    question = input("Enter question: ")
    from src.agent.nodes.prepare_discovery import extract_discovery_query
    parsed = extract_discovery_query(question)

    if not parsed.fields:
        print("No rankable field — prepare_discovery would block this question.")
    else:
        result = discovery_experiment({"discovery_query": parsed.model_dump()})
        print(result["tickers"])