"""
src/agent/formatters/valuation_formatter.py

Formats valuation_signal() results for the LLM — see
src/quant/valuation_signal.py for the data this consumes.
"""


def format_valuation(val: dict | None) -> str:
    """
    States the ratio comparison, who it was compared against (named
    peers, or the S&P 500 average if no peer data was available), and
    the resulting verdict — each stated exactly once.

    method tells you whether "ratio" is P/E or P/S — both methods
    return the same set of keys (see valuation_signal.py), so this
    formatter never needs to branch on which fields exist.
    """
    if not val:
        return "  Valuation: Insufficient data.\n"

    ratio_name = "P/E" if val["method"] == "pe" else "P/S"
    caveat = " — company is loss-making, not a directional call" if val.get("reference_only") else ""

    if val.get("peers_used"):
        comparison = f"peer median {val.get('benchmark_ratio')} (peers: {', '.join(val['peers_used'])})"
    else:
        comparison = f"S&P 500 average {val.get('benchmark_ratio')} (no peer-specific data available)"

    text  = "  Valuation:\n"
    text += f"    {val['valuation_label'].capitalize()}{caveat} (score={val['valuation_score']})\n"
    text += f"    {ratio_name} {val.get('ratio')} vs {comparison}\n"

    return text
