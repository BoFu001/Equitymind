from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    Internal state passed between LangGraph nodes.
    Each node reads from this state and writes back updates.
    """

    # Input
    question: str
    messages: list              # full conversation history
    session_memory: Optional[dict]  # structured + narrative summary memory

    # Macro classification (Layer 1)
    top_intent: Optional[str]   # TASK / GREETING / OUT_OF_SCOPE / GENERAL_KNOWLEDGE

    # Intent classification (Layer 2 — only runs if top_intent == TASK)
    sub_intent: Optional[str]   # SPECIFIC_STOCK / COMPARISON / DISCOVERY / ANALYZE_POSITION / ANALYZE_PORTFOLIO / FOLLOW_UP / CLARIFICATION

    clarification_complete: Optional[bool]  # True when clarification collected enough criteria
    enriched_query: Optional[str]  # transient — synthesized question for discovery_suggest, never persisted to messages

    # Extracted parameters
    tickers: Optional[list[str]]     # all tickers e.g. ["AAPL"] or ["AAPL", "MSFT"]
    year: Optional[str]         # e.g. "2025" or None for latest

    # Retrieval
    chunks: Optional[list]      # retrieved chunks from Pinecone

    # Market data
    market_data: Optional[dict] # price, P/E, revenue etc from yfinance

    # News and sentiment
    news: Optional[list]        # recent news articles with sentiment scores

    # Final output
    answer: Optional[str]       # final report


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
                "tickers_discussed":    [],
                "last_tickers":         [],
                "last_top_intent":      "",
                "last_sub_intent":      "",
                "last_market_data":     {}, 
                "in_clarification":     False, 
                "top_recommendations":  [],        # future
                "user_preferences": {              # future
                    "sectors": [],
                    "risk":    "",
                    "style":   "",
                }
            },
            "narrative": ""
        },
        "top_intent": None,
        "sub_intent": None,
        "clarification_complete": False,
        "enriched_query": None,
        "tickers":     [],
        "year":        None,
        "chunks":      [],
        "market_data": {},
        "news":        [],
        "answer":      "",
    }