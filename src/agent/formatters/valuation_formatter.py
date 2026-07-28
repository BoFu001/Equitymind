"""
src/agent/formatters/valuation_formatter.py

Formats valuation_signal() results for the LLM — see
src/quant/valuation_signal.py for the data this consumes.

Split out from the former single-file formatters.py on 2026-07-27,
alongside consensus_formatter.py — same principle: each signal's
formatter should live in its own file, one per data point.
"""


def format_valuation(val: dict | None) -> str:
    """
    Shows raw inputs (this ticker's own P/E and P/B, and the benchmark
    P/E and P/B being compared against) alongside the computed
    judgment, not just the judgment alone — a label like "overvalued
    (score=-0.86)" means little without knowing the actual P/E vs the
    benchmark P/E it was measured against (2026-07-27, same principle
    already applied to consensus_formatter.py).
    """
    if not val:
        return "  Valuation: Insufficient data.\n"

    text = "  Valuation:\n"

    # Tier 1 (P/E + P/B) exposes pe/pb/benchmark_pe/benchmark_pb as
    # independent fields; Tier 2 (P/S-only, reference_only=True) does
    # not have a peer benchmark_pe/pb to show (see valuation_signal.py
    # — Tier 2 uses a global P/S default, not a peer-specific figure).
    if val.get("benchmark_pe") is not None:
        text += (
            f"    Raw data: P/E={val.get('pe')} (benchmark {val.get('benchmark_pe')}), "
            f"P/B={val.get('pb')} (benchmark {val.get('benchmark_pb')}).\n"
        )

    text += f"    {val['valuation_label']} (score={val['valuation_score']}, method={val['method']})\n"
    text += f"    {val['detail']}\n"

    if val.get("peers_used"):
        text += f"    Compared against: {', '.join(val['peers_used'])}\n"
    if val.get("reference_only"):
        text += f"    Note: Valuation reference only — company is loss-making, P/S used instead of P/E.\n"
    if val.get("stale_benchmark"):
        text += f"    Note: Sector benchmarks may be outdated (>90 days old).\n"

    return text
