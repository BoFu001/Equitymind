"""
src/quant/valuation_signal.py

Valuation Signal Engine — Layer 2 Quantitative Intelligence.

Assesses whether a stock is overvalued or undervalued vs its peers,
using one of two independent methods depending on data availability:

    Method "pe" (profitable companies):  P/E vs peer P/E median
    Method "ps" (loss-making companies): P/S vs peer P/S median
    No data: Returns None — caller must handle gracefully

Both methods return the SAME set of keys (ratio, benchmark_ratio,
ratio_vs_peers, valuation_score, valuation_label, method,
reference_only, peers_used, detail) — method tells the caller which
ratio "ratio" actually is, so downstream code (report generation)
never needs to branch on which fields exist, only on what they mean.

Peer ratios are computed live from valuation_reader's peer_ratios
(each peer's own yfinance ratios, fetched at request time), not from
a precomputed batch snapshot. Extreme P/E values (outside 3-150) are
filtered before taking the median — loss-making or bubble-valuation
outliers are not representative comparison points.

If a company has positive earnings, P/E is used (the standard,
most-used valuation lens). If not, P/S is used instead — P/E is
mathematically undefined for a loss-making company, and professional
practice consistently substitutes P/S (or EV/Revenue) in this case,
not P/B (P/S reflects revenue, a more stable base than book value,
which can itself be distorted by buybacks or write-downs for a
struggling company).

Score range: -1.0 (severely overvalued) to +1.0 (severely undervalued),
             or None on the P/S path — a P/S discount and a P/E discount
             are different quantities and must not be ranked together.
Label:       "overvalued" / "fairly valued" / "undervalued" / "reference only" / None

Academic references:
    - P/E relative valuation: Damodaran (2012), Investment Valuation
    - P/S for loss-making firms: Fisher (1984), Super Stocks
"""

import statistics

DEFAULT_PE = 25.54  # S&P 500 P/E, GuruFocus, as of 2026-07-06
DEFAULT_PS = 3.70   # S&P 500 P/S, GuruFocus/S&P Dow Jones Indices, as of 2026-06-15


def _peer_median(peer_ratios: dict, key: str, lo: float, hi: float) -> tuple[float | None, list[str]]:
    """
    Computes the median of one ratio (key: "pe" or "ps") across peers,
    after filtering to (lo, hi) to exclude loss-making/extreme-outlier
    values. Returns (median, peers_used) — peers_used is the subset
    that actually passed the filter.
    """
    values = []
    used = []
    for symbol, ratios in peer_ratios.items():
        v = ratios.get(key)
        if isinstance(v, (int, float)) and lo < v < hi:
            values.append(v)
            used.append(symbol)
    if not values:
        return None, []
    return round(statistics.median(values), 2), sorted(used)


def _label(score: float | None, reference_only: bool) -> str:
    """Map numeric score to human-readable label."""
    if reference_only:
        return "reference only"
    if score > 0.2:
        return "undervalued"
    if score < -0.2:
        return "overvalued"
    return "fairly valued"


def _build_result(ratio: float, benchmark: float, peers_used: list[str],
                   method: str, ratio_name: str, reference_only: bool,
                   extra_detail: str = "") -> dict:
    """
    Shared result-builder for both methods — guarantees pe and ps
    paths return the identical set of keys, so downstream code never
    branches on which fields exist, only on what "method" says they mean.
    """
    # No score on the P/S path. The formula is the same, but its inputs
    # are not: a P/S discount against a peer P/S median is a different
    # quantity from a P/E discount against a peer P/E median, and the
    # two cannot be ranked against each other. The ratio and benchmark
    # below still answer "how is this company priced" on its own terms.
    if reference_only:
        score = None
    else:
        score = (benchmark - ratio) / benchmark
        score = round(max(-1.0, min(1.0, score)), 4)
    label = _label(score, reference_only)

    score_note = "" if score is None else f" (score={score})"
    peers_note = f" (compared against: {', '.join(peers_used)})" if peers_used else ""
    generic_note = "" if peers_used else " Comparison uses a broad S&P 500 average."

    return {
        "ratio":            ratio,
        "benchmark_ratio":  benchmark,
        "ratio_vs_peers":   f"{ratio} vs peer median {benchmark}",
        "valuation_score":  score,
        "valuation_label":  label,
        "method":           method,
        "reference_only":   reference_only,
        "peers_used":       peers_used,
        "detail": (
            f"{extra_detail.strip()} "
            f"{ratio_name} of {ratio} vs peer median {benchmark}{peers_note}"
            f"{score_note}. "
            f"→ {label}."
            f"{generic_note}"
        ).strip(),
    }


def valuation_signal(valuation_inputs: dict) -> dict | None:
    """
    Compute a valuation signal from valuation_reader.get_valuation_inputs().

    Uses P/E if the company is profitable, P/S if not. Returns None
    if neither is usable, allowing the report layer to skip valuation
    entirely and avoid hallucination.

    Args:
        valuation_inputs: dict from get_valuation_inputs(), expected fields:
            - pe_ratio:               float | None  (trailing P/E)
            - price_to_sales:         float | None  (P/S ratio)
            - ticker:                 str
            - peer_ratios:            dict — symbol -> {"pe": ..., "ps": ...}

    Returns:
        dict with keys (identical shape regardless of method):
            - ratio:             float — the ratio actually used (P/E or P/S)
            - benchmark_ratio:   float — peer median (or S&P 500 default)
            - ratio_vs_peers:    str
            - valuation_score:   float (-1.0 to +1.0), or None when method is "ps"
            - valuation_label:   str
            - method:            str — "pe" or "ps", tells you what "ratio" is
            - reference_only:    bool — True means P/S was used (loss-making company)
            - peers_used:        list[str]
            - detail:            str
        or None if no data available
    """

    pe          = valuation_inputs.get("pe_ratio")
    ps          = valuation_inputs.get("price_to_sales")
    peer_ratios = valuation_inputs.get("peer_ratios") or {}

    # Defensive type check: yfinance has, in practice, returned a
    # non-numeric string (e.g. "Infinity") for these ratio fields when
    # a company's EPS is near zero — a genuine mathematical edge case
    # of the P/E ratio, not a data error (confirmed for BILL,
    # 2026-07-24). Without this check, the comparisons below (pe > 0,
    # etc.) crash with a TypeError instead of gracefully degrading.
    if not isinstance(pe, (int, float)):
        pe = None
    if not isinstance(ps, (int, float)):
        ps = None

    if pe and pe > 0:
        pe_median, peers_used = _peer_median(peer_ratios, "pe", 3, 75)
        benchmark = pe_median if pe_median is not None else DEFAULT_PE
        return _build_result(pe, benchmark, peers_used, "pe", "P/E", reference_only=False)

    if ps and ps > 0:
        ps_median, peers_used = _peer_median(peer_ratios, "ps", 0, 100)
        benchmark = ps_median if ps_median is not None else DEFAULT_PS
        extra = " No usable P/E data available for this company (loss-making)."
        return _build_result(ps, benchmark, peers_used, "ps", "P/S", reference_only=True, extra_detail=extra)

    return None
