"""
src/agent/fetch_all_data.py

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

from src.tools.market_data import get_stock_snapshot, get_risk_inputs, get_quality_inputs, get_consensus_inputs
from src.tools.news_data import fetch_company_news
from src.tools.sec_retrieval import retrieve, fetch_embed_store_retrieve
from src.agent.state import AgentState

from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint, bprint


# ─────────────────────────────────────────────
# Stock snapshot
# ─────────────────────────────────────────────

def _fetch_stock_snapshot(ticker: str) -> dict | None:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["market_data_snapshot"].format(ticker=ticker)})

    data = get_stock_snapshot(ticker)
    if not data:
        return None

    bprint(f"  [_fetch_stock_snapshot] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Risk Signal inputs (2y price history, market benchmark, risk-free rate)
# ─────────────────────────────────────────────

def _fetch_risk_inputs(ticker: str) -> dict | None:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["market_data_risk_history"].format(ticker=ticker)})

    data = get_risk_inputs(ticker)
    bprint(f"  [_fetch_risk_inputs] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Quality Signal inputs (financial statements)
# ─────────────────────────────────────────────

def _fetch_quality_inputs(ticker: str) -> dict | None:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["market_data_financial_statements"].format(ticker=ticker)})

    data = get_quality_inputs(ticker)
    bprint(f"  [_fetch_quality_inputs] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# Consensus Signal inputs (analyst rating history)
# ─────────────────────────────────────────────

def _fetch_consensus_inputs(ticker: str) -> dict | None:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["market_data_analyst_ratings"].format(ticker=ticker)})

    data = get_consensus_inputs(ticker)
    bprint(f"  [_fetch_consensus_inputs] Fetched for {ticker}")
    return data


# ─────────────────────────────────────────────
# News Data
# ─────────────────────────────────────────────

def _fetch_news(ticker: str) -> list:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["news_data"].format(ticker=ticker)})

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
            writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["sec_retrieve"].format(ticker=ticker)})
        else:
            writer({"type": "sub_progress", "node": "fetch_all_data", "message": NODE_PROGRESS["sec_fetch"].format(ticker=ticker)})
            chunks = fetch_embed_store_retrieve(question, ticker)
    except Exception as e:
        bprint(f"  [_fetch_sec_data] Could not fetch SEC data for {ticker}: {e}")
        return []

    bprint(f"  [_fetch_sec_data] Fetched for {ticker}")
    return chunks


# ─────────────────────────────────────────────
# Main node: unconditional fetch for every ticker
# ─────────────────────────────────────────────

def fetch_all_data(state: AgentState) -> dict:
    """
    Fetches market data, news, and SEC filing excerpts for every ticker
    in state["tickers"], unconditionally and in a fixed order — no LLM
    judgement involved in deciding what to fetch.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "fetch_all_data", "message": NODE_PROGRESS["fetch_all_data"]})

    # Priority: enriched_query (clarification flow) > contextualized_question
    # (context-dependent follow-ups) > raw question (self-contained messages)
    question = state.get("enriched_query") or state.get("contextualized_question") or state["question"]
    gprint(f"  [fetch_all_data] question: {question}")
    tickers = state.get("tickers") or []

    all_stock_snapshots = {}
    all_risk            = {}
    all_quality         = {}
    all_consensus       = {}
    all_news            = {}
    all_chunks          = {}

    for ticker in tickers:
        # yfinance-sourced data grouped together first
        stock_snapshot = _fetch_stock_snapshot(ticker)
        if stock_snapshot:
            all_stock_snapshots[ticker] = stock_snapshot

        risk_inputs = _fetch_risk_inputs(ticker)
        if risk_inputs:
            all_risk[ticker] = risk_inputs

        quality_inputs = _fetch_quality_inputs(ticker)
        if quality_inputs:
            all_quality[ticker] = quality_inputs

        consensus_inputs = _fetch_consensus_inputs(ticker)
        if consensus_inputs:
            all_consensus[ticker] = consensus_inputs

        # Other data sources (finlight, SEC EDGAR)
        news_articles = _fetch_news(ticker)
        if news_articles:
            all_news[ticker] = news_articles

        sec_chunks = _fetch_sec_data(ticker, question)
        if sec_chunks:
            all_chunks[ticker] = sec_chunks

    gprint(f"  [fetch_all_data] Completed for {len(tickers)} ticker(s)")

    return {
        "stock_snapshots":   all_stock_snapshots,
        "risk_inputs":       all_risk,
        "quality_inputs":    all_quality,
        "consensus_inputs":  all_consensus,
        "news":              all_news,
        "chunks":            all_chunks,
    }
