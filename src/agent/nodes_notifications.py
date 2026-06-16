"""
nodes_notifications.py

Node progress and sub_progress messages for EquityMind.
Uses {placeholder} template pattern for variable injection.
Supports future internationalisation — swap NODE_PROGRESS dict for different language.
"""


# ─────────────────────────────────────────────
# Node progress messages — user-friendly UX language
# ─────────────────────────────────────────────
NODE_PROGRESS = {
    "classify":           "Understanding your question...",
    "extract":            "Identifying the company...",
    # ──────────────────────────────────────────────────────────────
    "ensure_sec_data":    "Checking our knowledge base...",
    "retrieve":           "Reading {ticker} annual report...",                    # sub_progress
    "fetch":              "Downloading {ticker} annual report from SEC...",       # sub_progress
    # ──────────────────────────────────────────────────────────────
    "market_data":        "Checking live market data...",
    "market_data_sub":    "Fetching live data for {ticker}...",                   # sub_progress
    # ──────────────────────────────────────────────────────────────
    "news":               "Reading the latest news...",
    "news_sub":           "Analysing news sentiment for {ticker}...",             # sub_progress
    # ──────────────────────────────────────────────────────────────
    "specific_report":    "Generating investment report...",
    #"comparison_report":  "Analysing and comparing companies...",
    "discovery_suggest":  "Searching for the best stocks for you...",
    #"discovery_report":   "Ranking and selecting top stocks...",
    "greeting":           "Welcome! Preparing your response...",
    "out_of_scope":       "Let me help you with that...",
    "no_ticker":          "Could not identify a stock ticker...",
    "follow_up":          "Looking up from our conversation...",
}