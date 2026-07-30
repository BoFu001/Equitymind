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
    "contextualize_question":              "Reviewing context...",
    "classify_top_intent":                 "Understanding your question...",
    "classify_sub_intent":                 "Identifying the task...",
    "greeting":                            "Welcome! Preparing your response...",
    "out_of_scope":                        "Let me help you with that...",
    "explain_concept":                     "Putting together an explanation...",
    "extract":                             "Pinpointing the company...",
    "determine_data_scope":                "Planning the analysis...",
    # ──────────────────────────────────────────────────────────────
    "fetch_all_data":                      "Collecting company data...",
    "sec_retrieve":                        "Reading {ticker} annual report...",                    # sub_progress
    "sec_fetch":                           "Downloading {ticker} annual report from SEC...",       # sub_progress
    "market_data_snapshot":                "Fetching {ticker}'s stock data...",                    # sub_progress
    "market_data_valuation":               "Checking {ticker}'s valuation ratios...",              # sub_progress
    "market_data_risk_history":            "Checking {ticker}'s risk history...",                  # sub_progress
    "market_data_financial_statements":    "Reviewing {ticker}'s financial statements...",         # sub_progress
    "financial_history":                   "Checking {ticker}'s historical financials...",         # sub_progress
    "market_data_analyst_ratings":         "Checking analyst ratings for {ticker}...",             # sub_progress
    "news_data":                           "Reading news for {ticker}...",                         # sub_progress
    # ──────────────────────────────────────────────────────────────
    "generate_report":                     "Generating investment report...",
    "discovery_suggest":                   "Searching for the best stocks for you...",
    "no_ticker":                           "Could not identify a stock ticker...",
    # ──────────────────────────────────────────────────────────────
    "clarification":                       "Reviewing your investment criteria...",
    "clarification_sub":                   "Gathering more criteria...",                           # sub_progress
    # ──────────────────────────────────────────────────────────────
    "quant_engine":                        "Running quantitative signals...",
    "quant_valuation":                     "Computing valuation signal for {ticker}...",           # sub_progress
    "quant_momentum":                      "Computing momentum signal for {ticker}...",            # sub_progress
    "quant_risk":                          "Computing risk signal for {ticker}...",                # sub_progress
    "quant_quality":                       "Computing quality signal for {ticker}...",             # sub_progress
    "quant_news_sentiment":                "Computing news sentiment signal for {ticker}...",      # sub_progress
    "quant_consensus":                     "Computing consensus signal for {ticker}...",           # sub_progress
}