"""
src/agent/formatters/financial_history_formatter.py

Formats multi-year financial_history rows (see
src/tools/financial_history_reader.py) into a plain-text
year-over-year table for the LLM.

Split out from the former single-file formatters.py on 2026-07-27,
alongside the per-signal formatters. Not tied to any quant signal —
this displays raw historical financials for trend questions
(e.g. "Apple's revenue over the last few years").

Shows total_revenue and net_income only (the two figures users most
commonly ask about) — not all 26 stored metrics, to avoid overloading
the prompt with numbers the question didn't ask about.
"""


def format_financial_history(rows: list, ticker: str) -> str:
    if not rows:
        return f"\n{ticker}: No historical financial data available.\n"

    annual_rows = [r for r in rows if r["period_type"] == "annual"]
    quarterly_rows = [r for r in rows if r["period_type"] == "quarterly"]

    text = f"\n{ticker} Historical Financials:\n"

    def _format_row(r):
        rev = r.get("total_revenue")
        ni = r.get("net_income")
        if rev is not None and ni is not None:
            return f"    {r['period_end']}: Revenue ${rev/1e9:.2f}B, Net Income ${ni/1e9:.2f}B\n"
        return f"    {r['period_end']}: Revenue/Net Income data incomplete\n"

    if annual_rows:
        text += "  Annual (fiscal year end):\n"
        for r in sorted(annual_rows, key=lambda x: x["period_end"]):
            text += _format_row(r)

    if quarterly_rows:
        text += "  Quarterly (most recent periods):\n"
        for r in sorted(quarterly_rows, key=lambda x: x["period_end"]):
            text += _format_row(r)

    return text
