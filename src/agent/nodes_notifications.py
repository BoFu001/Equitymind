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
    "fetch_data":                          "Collecting company data...",
    "snapshot_fetch":                      "Fetching {ticker}'s stock data...",                    # sub_progress
    "sec_retrieve":                        "Reading {ticker} annual report...",                    # sub_progress
    "sec_fetch":                           "Downloading {ticker} annual report from SEC...",       # sub_progress
    "valuation_fetch":                     "Checking {ticker}'s valuation ratios...",              # sub_progress
    "momentum_fetch":                      "Checking {ticker}'s momentum data...",                 # sub_progress
    "risk_fetch":                          "Checking {ticker}'s risk history...",                  # sub_progress
    "quality_fetch":                       "Reviewing {ticker}'s financial statements...",         # sub_progress
    "consensus_fetch":                     "Checking analyst ratings for {ticker}...",             # sub_progress
    "news_data":                           "Reading news for {ticker}...",                         # sub_progress
    "financial_history":                   "Checking {ticker}'s historical financials...",         # sub_progress
    "short_fetch":                         "Checking {ticker}'s short interest...",                # sub_progress
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
    "quant_short":                         "Computing short signal for {ticker}...",               # sub_progress
}