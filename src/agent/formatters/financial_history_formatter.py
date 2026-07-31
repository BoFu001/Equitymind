"""
src/agent/formatters/financial_history_formatter.py

Formats multi-year financial_history rows (see
src/readers/financial_history_reader.py) into a plain-text
year-over-year table for the LLM.

Split out from the former single-file formatters.py on 2026-07-27,
alongside the per-signal formatters. Not tied to any quant signal —
this displays raw historical financials for trend questions
(e.g. "Apple's revenue over the last few years").

Shows ALL 26 stored metrics (2026-07-27 revision) — an earlier version
of this function only showed total_revenue and net_income, on the
assumption that users only care about those two figures. That
assumption was never actually confirmed and doesn't match the
"expose all available raw data" standard applied to every other
signal's formatter that day (consensus_formatter.py's analyst_count/
latest_rating_counts, valuation_formatter.py's benchmark_pe/benchmark_pb,
risk_formatter.py's raw_beta/peak_price/trough_price, etc.) — the
financial_history table already stores all 26 metrics precisely so
this kind of trend question can be answered in full, and withholding
24 of them was an unreviewed, one-sided decision, not a deliberate
design choice.
"""


def _fmt(value, unit="dollars"):
    """
    Formats a raw metric value for display.
    unit="dollars": large dollar figures, shown as $X.XXB.
    unit="count": large non-dollar counts (e.g. shares_outstanding),
        shown as X.XXB with no dollar sign — this is a share count,
        not a monetary amount, and labeling it with "$" would be
        misleading (2026-07-27 fix).
    unit="raw": small figures shown as-is (e.g. EPS).
    """
    if value is None:
        return "N/A"
    if unit == "dollars":
        return f"${value/1e9:.2f}B"
    if unit == "count":
        return f"{value/1e9:.2f}B"
    return f"{value:.2f}"


def _format_row(r: dict) -> str:
    period = r["period_end"]
    lines = [f"    {period}:"]

    lines.append(
        f"      Income Statement — Revenue: {_fmt(r.get('total_revenue'))}, "
        f"Cost of Revenue: {_fmt(r.get('cost_of_revenue'))}, "
        f"Gross Profit: {_fmt(r.get('gross_profit'))}, "
        f"R&D: {_fmt(r.get('research_and_development'))}, "
        f"SG&A: {_fmt(r.get('selling_general_and_administration'))}, "
        f"Operating Expense: {_fmt(r.get('operating_expense'))}, "
        f"Operating Income: {_fmt(r.get('operating_income'))}, "
        f"EBIT: {_fmt(r.get('ebit'))}, "
        f"EBITDA: {_fmt(r.get('ebitda'))}, "
        f"Pretax Income: {_fmt(r.get('pretax_income'))}, "
        f"Net Income: {_fmt(r.get('net_income'))}, "
        f"Diluted EPS: {_fmt(r.get('diluted_eps'), unit='raw')}, "
        f"Basic EPS: {_fmt(r.get('basic_eps'), unit='raw')}"
    )
    lines.append(
        f"      Cash Flow — Operating CF: {_fmt(r.get('operating_cash_flow'))}, "
        f"CapEx: {_fmt(r.get('capital_expenditure'))}, "
        f"Free Cash Flow: {_fmt(r.get('free_cash_flow'))}, "
        f"Stock Buybacks: {_fmt(r.get('repurchase_of_capital_stock'))}, "
        f"Dividends Paid: {_fmt(r.get('cash_dividends_paid'))}"
    )
    lines.append(
        f"      Balance Sheet — Total Assets: {_fmt(r.get('total_assets'))}, "
        f"Total Liabilities: {_fmt(r.get('total_liabilities'))}, "
        f"Stockholders' Equity: {_fmt(r.get('stockholders_equity'))}, "
        f"Cash & Equivalents: {_fmt(r.get('cash_and_equivalents'))}, "
        f"Long-Term Debt: {_fmt(r.get('long_term_debt'))}, "
        f"Current Assets: {_fmt(r.get('current_assets'))}, "
        f"Current Liabilities: {_fmt(r.get('current_liabilities'))}, "
        f"Shares Outstanding: {_fmt(r.get('shares_outstanding'), unit='count')} shares"
    )
    return "\n".join(lines) + "\n"


def format_financial_history(rows: list, ticker: str) -> str:
    if not rows:
        return f"\n{ticker}: No historical financial data available.\n"

    annual_rows = [r for r in rows if r["period_type"] == "annual"]
    quarterly_rows = [r for r in rows if r["period_type"] == "quarterly"]

    # Explicit coverage range, stated up front — added 2026-07-27 after
    # real testing showed the LLM would answer questions about years
    # outside this range (e.g. "Ford's EBITDA in 2018" when the data
    # here only goes back to 2022) using its own training knowledge
    # instead of stating the data doesn't cover that year. Listing the
    # years alone wasn't enough to prevent this — an explicit "coverage
    # is X to Y" statement, paired with a prompt-level rule (see
    # generate_report.py), is needed to make the boundary unambiguous.
    all_period_ends = [r["period_end"] for r in rows]
    earliest = min(all_period_ends)
    latest = max(all_period_ends)

    text = f"\n{ticker} Historical Financials (DATA COVERAGE: {earliest} to {latest} ONLY — do not answer questions about periods outside this range using outside knowledge; state that the data doesn't cover that period instead):\n"

    if annual_rows:
        text += "  Annual (fiscal year end):\n"
        for r in sorted(annual_rows, key=lambda x: x["period_end"]):
            text += _format_row(r)

    if quarterly_rows:
        text += "  Quarterly (most recent periods):\n"
        for r in sorted(quarterly_rows, key=lambda x: x["period_end"]):
            text += _format_row(r)

    return text
