"""
src/agent/formatters/consensus_formatter.py

Formats consensus_signal() results for the LLM — see
src/quant/consensus_signal.py for the data this consumes.

Split out from the former single-file formatters.py on 2026-07-27 —
first of the format_xxx() functions to be moved into its own file, one
per data point, matching the same principle already applied to
src/readers/ (snapshot_reader.py, risk_reader.py, consensus_reader.py,
quality_reader.py). The rest of formatters.py's functions remain in
__init__.py for now and will be moved out one at a time as each
signal's data flow gets reviewed — see project notes, 2026-07-27.
"""


def format_consensus(consensus: dict | None) -> str:
    """
    Shows the raw analyst data and nothing derived from it: the 1-5
    recommendation mean with its scale spelled out, the target price
    range and how far the mean target sits above the current price, and
    how many analysts sit in each rating bucket. What Discovery sorts on
    is the same recommendation mean printed here.
    """
    if not consensus:
        return "  Consensus: Insufficient data.\n"
    text = "  Consensus (Analyst Opinions):\n"
    text += (
        f"    Raw data: recommendation_mean={consensus['recommendation_mean']} "
        f"(1=Strong Buy, 5=Strong Sell), target price range "
        f"${consensus['target_low']}-${consensus['target_high']} "
        f"(mean ${consensus['target_mean']}, {consensus['upside_pct']}% "
        f"above the current ${consensus['current_price']}), based on "
        f"{consensus['analyst_count']} analysts.\n"
    )
    if consensus.get("latest_rating_counts"):
        rc = consensus["latest_rating_counts"]
        text += (
            f"    Current analyst split: {rc['strongBuy']} Strong Buy, "
            f"{rc['buy']} Buy, {rc['hold']} Hold, {rc['sell']} Sell, "
            f"{rc['strongSell']} Strong Sell.\n"
        )
    text += f"  {consensus['detail']}\n"
    if consensus.get("low_confidence"):
        text += f"  Note: Low analyst sample size ({consensus['analyst_count']} analysts) — reduced confidence.\n"
    if consensus.get("wide_dispersion"):
        text += f"  Note: Wide target price dispersion (${consensus['target_low']} to ${consensus['target_high']}) — significant analyst disagreement.\n"
    return text
