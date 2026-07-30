"""
src/agent/fetch_data.py

Data Acquisition Node — fetches all data needed for downstream signal
computation (quant_engine) and report generation (generate_report).

No LLM calls — pure, deterministic data fetching. For every ticker
identified in the question, unconditionally fetches market data, news,
and SEC filing excerpts from all three underlying data sources
(yfinance, finlight, SEC EDGAR).

This replaced an earlier "smart tool selection" design where an LLM
agent loop decided which data sources to call per question. That design
was removed because: (1) quant_engine's signal engines depend on these
same three data sources regardless of question phrasing — Valuation,
Risk, Quality, and Consensus all require market_data internally, so
letting an LLM's tool-selection judgement skip market_data caused
quant_engine's blanket "no market_data available" check to skip ALL
signals, including ones (News Sentiment) that didn't actually need the
skipped data; and (2) the three data sources are cheap enough (free or
low-cost APIs) that unconditional fetching costs little relative to the
reliability gained from removing an unpredictable LLM decision point.
"""

import asyncio

from src.readers.snapshot_reader import get_stock_snapshot
from src.readers.valuation_reader import get_valuation_inputs
from src.readers.risk_reader import get_risk_inputs
from src.readers.consensus_reader import get_consensus_snapshot, get_consensus_trend
from src.readers.quality_reader import get_quality_inputs_from_db
from src.readers.news_reader import fetch_company_news
from src.readers.sec_retrieval import retrieve, fetch_embed_store_retrieve
from src.readers.financial_history_reader import get_financial_history_rows
from src.agent.nodes.determine_data_scope import VALID_SIGNALS
from src.agent.state import AgentState

from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint, bprint


# ─────────────────────────────────────────────
# Stock snapshot
# ─────────────────────────────────────────────

def _fetch_stock_snapshot(ticker: str) -> dict | None:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["market_data_snapshot"].format(ticker=ticker)})

    data = get_stock_snapshot(ticker)
    if not data:
        return None

    bprint(f"  [_fetch_stock_snapshot] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Valuation Signal inputs (pe_ratio, price_to_book, price_to_sales)
# ─────────────────────────────────────────────

def _fetch_valuation_inputs(ticker: str) -> dict | None:
    """
    Fetched independently of snapshot_reader.py as of 2026-07-27 —
    see valuation_reader.py for why (same principle already applied
    to consensus_reader.py: every signal's data point should be
    independently fetchable).
    """
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["market_data_valuation"].format(ticker=ticker)})

    data = get_valuation_inputs(ticker)
    bprint(f"  [_fetch_valuation_inputs] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Risk Signal inputs (2y price history, market benchmark, risk-free rate)
# ─────────────────────────────────────────────

def _fetch_risk_inputs(ticker: str) -> dict | None:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["market_data_risk_history"].format(ticker=ticker)})

    data = get_risk_inputs(ticker)
    bprint(f"  [_fetch_risk_inputs] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Quality Signal inputs (financial statements)
# ─────────────────────────────────────────────

def _fetch_quality_inputs(ticker: str) -> dict | None:
    # Reads from financial_history (DB) instead of calling yfinance
    # live — see src/tools/quality_reader.py for why, and
    # why there is deliberately no live-yfinance fallback here.
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["market_data_financial_statements"].format(ticker=ticker)})

    data = get_quality_inputs_from_db(ticker)
    bprint(f"  [_fetch_quality_inputs] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Consensus Signal inputs (analyst rating history)
# ─────────────────────────────────────────────

def _fetch_consensus_inputs(ticker: str) -> dict | None:
    """
    Fetches BOTH pieces consensus_signal() needs — the point-in-time
    snapshot (recommendation_mean, target_mean, etc.) and the rating
    trend history — via two independent calls (consensus_reader.py),
    not shared with snapshot_reader.py. See consensus_reader.py's
    module docstring for why this duplicates part of what
    _fetch_stock_snapshot() also fetches.

    Returns None only if the snapshot portion fails — consensus_signal()
    cannot compute anything without it. The trend portion is allowed to
    be None independently (consensus_signal() degrades gracefully).
    """
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["market_data_analyst_ratings"].format(ticker=ticker)})

    snapshot = get_consensus_snapshot(ticker)
    if not snapshot:
        return None

    trend = get_consensus_trend(ticker)

    bprint(f"  [_fetch_consensus_inputs] Fetched for {ticker}")
    return {"snapshot": snapshot, "trend": trend}


# ─────────────────────────────────────────────
# Historical Financials (multi-year revenue/income/balance-sheet trend)
# ─────────────────────────────────────────────

def _fetch_financial_history(ticker: str) -> list:
    """
    Moved here from generate_report.py on 2026-07-27 — a report
    generation node calling a database reader directly violated the
    project's data-layer/compute-layer/display-layer separation
    (fetch_data owns all data acquisition; generate_report should
    only consume already-fetched state and format it). Also makes this
    signal controllable by determine_data_scope's signals_needed, the
    same way as valuation/risk/quality/consensus/news.
    """
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["financial_history"].format(ticker=ticker)})

    rows = get_financial_history_rows(ticker)
    bprint(f"  [_fetch_financial_history] Fetched for {ticker}")
    return rows


# ─────────────────────────────────────────────
# News Data
# ─────────────────────────────────────────────

def _fetch_news(ticker: str) -> list:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["news_data"].format(ticker=ticker)})

    articles = fetch_company_news(ticker)
    bprint(f"  [_fetch_news] Fetched for {ticker}")
    return articles


# ─────────────────────────────────────────────
# SEC Filing Retrieval
# ─────────────────────────────────────────────

def _fetch_sec_data(ticker: str, question: str) -> list:
    writer = get_stream_writer()

    try:
        chunks = retrieve(question, ticker)
        if chunks:
            writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["sec_retrieve"].format(ticker=ticker)})
        else:
            writer({"type": "sub_progress", "node": "fetch_data", "message": NODE_PROGRESS["sec_fetch"].format(ticker=ticker)})
            chunks = fetch_embed_store_retrieve(question, ticker)
    except Exception as e:
        bprint(f"  [_fetch_sec_data] Could not fetch SEC data for {ticker}: {e}")
        return []

    bprint(f"  [_fetch_sec_data] Fetched for {ticker}")
    return chunks


# ─────────────────────────────────────────────
# Main node: unconditional fetch for every ticker
# ─────────────────────────────────────────────

async def fetch_data(state: AgentState) -> dict:
    """
    Fetches only the data sources determine_data_scope decided this
    question needs (state["signals_needed"]), plus snapshot and SEC
    filing data which are always fetched regardless (basic company
    info and filing excerpts any question may reference).

    Falls back to fetching ALL signals if signals_needed is missing
    or empty (e.g. determine_data_scope failed) — a full-analysis
    default is the safe degradation, matching that node's own
    fallback behavior.

    Uses a dict of {name: coroutine} rather than a fixed-position
    tuple unpack from asyncio.gather, since the set of tasks now
    varies per request — a fixed unpack would break as soon as the
    number of tasks differs from one call to the next.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "fetch_data", "message": NODE_PROGRESS["fetch_data"]})

    # Priority: enriched_query (clarification flow) > contextualized_question
    # (context-dependent follow-ups) > raw question (self-contained messages)
    question = state.get("enriched_query") or state.get("contextualized_question") or state["question"]
    gprint(f"  [fetch_data] question: {question}")
    tickers = state.get("tickers") or []
    signals_needed = state.get("signals_needed") or list(VALID_SIGNALS)
    gprint(f"  [fetch_data] signals_needed: {signals_needed}")

    all_stock_snapshots   = {}
    all_valuation         = {}
    all_risk              = {}
    all_quality           = {}
    all_consensus         = {}
    all_news              = {}
    all_chunks            = {}
    all_financial_history = {}

    for ticker in tickers:
        tasks = {
            "snapshot": asyncio.to_thread(_fetch_stock_snapshot, ticker),
            "sec":      asyncio.to_thread(_fetch_sec_data, ticker, question),
        }
        if "valuation" in signals_needed:
            tasks["valuation"] = asyncio.to_thread(_fetch_valuation_inputs, ticker)
        if "risk" in signals_needed:
            tasks["risk"] = asyncio.to_thread(_fetch_risk_inputs, ticker)
        if "quality" in signals_needed:
            tasks["quality"] = asyncio.to_thread(_fetch_quality_inputs, ticker)
        if "consensus" in signals_needed:
            tasks["consensus"] = asyncio.to_thread(_fetch_consensus_inputs, ticker)
        if "news" in signals_needed:
            tasks["news"] = asyncio.to_thread(_fetch_news, ticker)
        if "financial_history" in signals_needed:
            tasks["financial_history"] = asyncio.to_thread(_fetch_financial_history, ticker)

        results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

        if results.get("snapshot"):
            all_stock_snapshots[ticker] = results["snapshot"]
        if results.get("valuation"):
            all_valuation[ticker] = results["valuation"]
        if results.get("risk"):
            all_risk[ticker] = results["risk"]
        if results.get("quality"):
            all_quality[ticker] = results["quality"]
        if results.get("consensus"):
            all_consensus[ticker] = results["consensus"]
        if results.get("news"):
            all_news[ticker] = results["news"]
        if results.get("sec"):
            all_chunks[ticker] = results["sec"]
        if results.get("financial_history"):
            all_financial_history[ticker] = results["financial_history"]

    gprint(f"  [fetch_data] Completed for {len(tickers)} ticker(s)")

    return {
        "stock_snapshots":        all_stock_snapshots,
        "valuation_inputs":       all_valuation,
        "risk_inputs":            all_risk,
        "quality_inputs":         all_quality,
        "consensus_inputs":       all_consensus,
        "news":                   all_news,
        "chunks":                 all_chunks,
        "financial_history_data": all_financial_history,
    }
