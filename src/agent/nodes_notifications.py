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
    "classify_top_intent":         "Understanding your question...",
    "classify_sub_intent":         "Identifying the task...",
    "greeting":                    "Welcome! Preparing your response...",
    "out_of_scope":                "Let me help you with that...",
    "explain_concept":             "Putting together an explanation...",
    "extract":                     "Pinpointing the company...",
    "research_loop":               "Planning the research...",
    # ──────────────────────────────────────────────────────────────
    "ensure_sec_data":             "Checking our knowledge base...",
    "retrieve":                    "Reading {ticker} annual report...",                    # sub_progress
    "fetch":                       "Downloading {ticker} annual report from SEC...",       # sub_progress
    # ──────────────────────────────────────────────────────────────
    "market_data":                 "Checking live market data...",
    "market_data_sub":             "Fetching live data for {ticker}...",                   # sub_progress
    # ──────────────────────────────────────────────────────────────
    "news":                        "Reading the latest news...",
    "news_sub":                    "Analysing news sentiment for {ticker}...",             # sub_progress
    # ──────────────────────────────────────────────────────────────
    "generate_report":             "Generating investment report...",
    "discovery_suggest":           "Searching for the best stocks for you...",
    "no_ticker":                   "Could not identify a stock ticker...",
    "follow_up":                   "Looking up from our conversation...",
    # ──────────────────────────────────────────────────────────────
    "clarification":               "Reviewing your investment criteria...",
    "clarification_sub":           "Gathering more criteria...",                           # sub_progress
}