from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    Internal state passed between LangGraph nodes.
    Each node reads from this state and writes back updates.
    """

    # Input
    question: str
    messages: list                          # full conversation history
    session_memory: Optional[dict]          # structured + narrative summary memory

    # Contextualization — runs before Layer 1 classify_top_intent
    contextualized_question: Optional[str]  # question rewritten to be self-contained, or same as question if no rewrite needed

    # Macro classification (Layer 1)
    top_intent: Optional[str]               # TASK / GREETING / OUT_OF_SCOPE / GENERAL_KNOWLEDGE

    # Intent classification (Layer 2 — only runs if top_intent == TASK)
    sub_intent: Optional[str]               # SPECIFIC_STOCK / COMPARISON / DISCOVERY / ANALYZE_POSITION / ANALYZE_PORTFOLIO / CLARIFICATION

    clarification_complete: Optional[bool]  # True when clarification collected enough criteria
    enriched_query: Optional[str]           # transient — synthesized question for discovery_suggest, never persisted to messages

    # Extracted parameters
    tickers: Optional[list[str]]            # all tickers e.g. ["AAPL"] or ["AAPL", "MSFT"]
    year: Optional[str]                     # e.g. "2025" or None for latest    

    # Which data sources/signals this question actually needs — decided once,
    # after tickers are confirmed, before fetch_data. snapshot is always
    # fetched regardless (basic company info any question may reference) and
    # is not part of this list. See determine_data_scope.py.
    signals_needed: Optional[list[str]]     # subset of: valuation, momentum, risk, quality, consensus, news, financial_history
    
    # News and sentiment
    news: Optional[list]                    # recent news articles with sentiment scores

    # Retrieval
    chunks: Optional[list]                  # retrieved chunks from pgvector

    # Stock snapshots
    stock_snapshots: Optional[dict]         # price, P/E, revenue etc from yfinance
    valuation_inputs: Optional[dict]        # pe_ratio, price_to_book, price_to_sales — for Valuation Signal

    # Inputs for quant_engine signal calculations (fetched by fetch_data,
    # consumed by quant_engine — quant_engine performs no I/O of its own)
    risk_inputs: Optional[dict]             # price history, market benchmark, risk-free rate — for Risk Signal
    quality_inputs: Optional[dict]          # financial statements — for Quality Signal
    consensus_inputs: Optional[dict]        # analyst rating history — for Consensus Signal
    financial_history_data: Optional[dict]  # multi-year financials (all 26 metrics) — for HISTORICAL FINANCIALS prompt section, fetched by fetch_data (moved from generate_report.py 2026-07-27, see fetch_data.py)

    # Quantitative signals — Layer 2
    quant_signals: Optional[dict]           # computed by quant_engine node

    # Final output
    answer: Optional[str]                   # final report


def build_initial_state(question: str, messages: list | None = None, session_memory: dict | None = None) -> dict:
    """
    Builds the initial AgentState dict for a new request.

    Usage 1 — WebSocket streaming endpoint:
        initial_state = build_initial_state(question)
        await graph.astream(initial_state, stream_mode="updates")

    Usage 2 — Sync REST endpoint:
        initial_state = build_initial_state(request.question)
        final_state = await graph.ainvoke(initial_state)

    All fields start empty except question.
    Each node fills its own fields as the graph executes.
    """
    return {
        "question":    question,
        "messages":    messages or [],
        "session_memory": session_memory or {
            "structured": {
                "last_tickers":         [],
                "in_clarification":     False,
            },
            "narrative": ""
        },
        "contextualized_question": None,
        "top_intent": None,
        "sub_intent": None,
        "clarification_complete": False,
        "enriched_query": None,
        "tickers":     [],
        "year":        None,
        "signals_needed": [],
        "news":        [],
        "chunks":      [],
        "stock_snapshots": {},
        "valuation_inputs": {},
        "risk_inputs": {},
        "quality_inputs": {},
        "consensus_inputs": {},
        "financial_history_data": {},
        "quant_signals": {},
        "answer":      "",
    }