"""
src/agent/formatters/momentum_formatter.py

Formats momentum_signal() results for the LLM — see
src/quant/momentum_signal.py for the data this consumes.

Split out from the former single-file formatters.py on 2026-07-27,
alongside consensus_formatter.py, valuation_formatter.py, and
snapshot_formatter.py. No content changes — this signal's raw inputs
(historical daily closing prices) aren't meaningful to display
directly (unlike Consensus/Valuation's simple numeric ratios), so
momentum_12_1_pct and position_52w_percentile already ARE the
appropriately-compressed representation of the underlying price
history, not a summary that omits something displayable (reviewed
2026-07-27, no gap found).
"""


def format_momentum(mom: dict | None) -> str:
    """
    Two independent sub-signals, NOT combined into a single score
    (12-1 momentum answers "how much has it moved over the past year",
    52-week position answers "how close is it to its own high right
    now" — different questions, not to be averaged).
    """
    if not mom:
        return "  Momentum: Insufficient data (e.g. recent IPO with limited price history).\n"
    text = f"  Momentum - 12-1 Month Return: {mom['momentum_12_1_label']} ({mom['momentum_12_1_pct']}%, percentile={mom['momentum_12_1_percentile']})\n"
    text += f"  Momentum - 52-Week High Position: {mom['position_52w_label']} ({round(mom['position_52w']*100)}% of 52w range, percentile={mom['position_52w_percentile']})\n"
    text += f"  {mom['detail']}\n"
    if mom.get("stale_benchmark"):
        text += f"  Note: Momentum benchmarks may be outdated (>90 days old).\n"
    return text
