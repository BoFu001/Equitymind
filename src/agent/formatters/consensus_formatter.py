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
    Three independent sub-signals, NOT combined into a single score
    (recommendation/upside/trend answer different questions: current
    standing, future price target, and recent directional change —
    averaging them would hide the full picture).

    Shows raw inputs (recommendation_mean, target price range,
    analyst_count) alongside the computed judgments, not just the
    judgments alone.
    """
    if not consensus:
        return "  Consensus: Insufficient data.\n"
    text = "  Consensus (Analyst Opinions):\n"
    text += (
        f"    Raw data: recommendation_mean={consensus['recommendation_mean']} "
        f"(1=Strong Buy, 5=Strong Sell), target price range "
        f"${consensus['target_low']}-${consensus['target_high']} "
        f"(mean ${consensus['target_mean']}), based on "
        f"{consensus['analyst_count']} analysts.\n"
    )
    if consensus.get("latest_rating_counts"):
        rc = consensus["latest_rating_counts"]
        text += (
            f"    Current analyst split: {rc['strongBuy']} Strong Buy, "
            f"{rc['buy']} Buy, {rc['hold']} Hold, {rc['sell']} Sell, "
            f"{rc['strongSell']} Strong Sell.\n"
        )
    text += f"  Consensus - Recommendation: {consensus['recommendation_label']} (score={consensus['recommendation_score']})\n"
    text += f"  Consensus - Upside: {consensus['upside_label']} ({consensus['upside_pct']}% implied by analyst target price)\n"
    if consensus.get("trend_label") is not None:
        text += f"  Consensus - Trend: {consensus['trend_label']} (trend_score={consensus['trend_score']})\n"
    else:
        text += f"  Consensus - Trend: Insufficient rating history.\n"
    text += f"  {consensus['detail']}\n"
    if consensus.get("low_confidence"):
        text += f"  Note: Low analyst sample size ({consensus['analyst_count']} analysts) — reduced confidence.\n"
    if consensus.get("wide_dispersion"):
        text += f"  Note: Wide target price dispersion (${consensus['target_low']} to ${consensus['target_high']}) — significant analyst disagreement.\n"
    return text
