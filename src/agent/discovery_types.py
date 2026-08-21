"""
src/agent/discovery_types.py

Shared vocabulary for Discovery: the fields a user can rank by, and the
structured form of a parsed question.

Lives above nodes/ because two nodes depend on it and neither owns it.
discovery_preparation parses a question into a DiscoveryQuery; discovery
executes one. Keeping the definitions here means the parser and the
executor cannot drift apart, and neither has to import the other.

The three field lists, VALID_FIELD_NAMES and RankField are one unit:
VALID_FIELD_NAMES is built from the lists and RankField.name is typed by
it, so splitting them across modules would only create a cycle.
"""

from typing import Literal

from pydantic import BaseModel

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
