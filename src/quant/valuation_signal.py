"""
src/quant/valuation_signal.py

Valuation Signal Engine — Layer 2 Quantitative Intelligence.

Assesses whether a stock is overvalued or undervalued using a two-tier
degradation strategy based on data availability:

    Tier 1 (most reliable):  P/E + P/B (vs peer group)     → method = "pe_pb"
    Tier 2 (reference only): P/S ratio (loss-making firms) → method = "ps_only"
    Tier 3 (no data):        Returns None — caller must handle gracefully

P/E and P/B are independent valuation lenses (earnings-based vs
book-value-based) — combined via a 0.6/0.4 weighted score. PEG was
removed: it is mathematically derived from P/E (PEG = P/E / growth),
so combining it with P/E double-counts the same underlying information
rather than adding independent evidence — a pattern professional
multi-factor value models (e.g. AQR) avoid, generally preferring
independent ratios (P/E, P/B, EV/Sales, cash flow yield) over PEG.

Analyst target price ("upside") was also removed from this signal — it
reflects human analyst judgment, not an objective market multiple, and
now lives in the separate Consensus Signal, keeping Valuation a purely
mechanical, objective measure.

Score range: -1.0 (severely overvalued) to +1.0 (severely undervalued)
Label:       "overvalued" / "fairly valued" / "undervalued" / "reference only" / None

Academic references:
    - P/E and P/B relative valuation: Damodaran (2012), Investment Valuation
    - Independent multi-factor value composites: Asness, Frazzini & Pedersen,
      "Quality Minus Junk" (AQR); Fama & French value factor methodology
    - P/S for loss-making firms: Fisher (1984), Super Stocks
"""


import json
from pathlib import Path
from datetime import date

# ─────────────────────────────────────────────
# Load valuation benchmarks from JSON
# Updated daily by scripts/update_valuation_benchmarks.py — peer
# IDENTITY (which companies) comes from peer_groups.json instead,
# refreshed separately/infrequently by scripts/update_peer_groups.py
# (see that script's docstring for why these were split).
# ─────────────────────────────────────────────
_VALUATION_BENCHMARKS_PATH = Path(__file__).parent / "data" / "valuation_benchmarks.json"

def _load_valuation_benchmarks() -> dict:
    """Load valuation benchmarks from JSON file."""
    with open(_VALUATION_BENCHMARKS_PATH) as f:
        return json.load(f)

_VALUATION_BENCHMARKS = _load_valuation_benchmarks()
_BENCHMARKS_DATA = _VALUATION_BENCHMARKS.get("benchmarks", {})
DEFAULT_PE       = _VALUATION_BENCHMARKS.get("default_pe", 25.54)
DEFAULT_PB       = _VALUATION_BENCHMARKS.get("default_pb", 5.44)
DEFAULT_PS       = _VALUATION_BENCHMARKS.get("default_ps", 3.70)  

# ── Staleness check ──────────────────────────
def _benchmarks_are_stale() -> bool:
    """Warn if benchmarks are more than 90 days old."""
    updated_at = _VALUATION_BENCHMARKS.get("updated_at", "")
    if not updated_at:
        return True
    try:
        updated = date.fromisoformat(updated_at)
        return (date.today() - updated).days > 90
    except ValueError:
        return True

BENCHMARKS_STALE = _benchmarks_are_stale()


def _get_benchmark_pe(ticker: str) -> float:
    """
    Look up benchmark P/E for a ticker from valuation_benchmarks.json.
    Falls back to DEFAULT_PE if ticker not found.
    """
    entry = _BENCHMARKS_DATA.get(ticker)
    if entry:
        return entry["benchmark_pe"]
    return DEFAULT_PE


def _get_benchmark_pb(ticker: str) -> float:
    """
    Look up benchmark P/B for a ticker from valuation_benchmarks.json.
    Falls back to DEFAULT_PB if ticker not found.
    """
    entry = _BENCHMARKS_DATA.get(ticker)
    if entry:
        return entry["benchmark_pb"]
    return DEFAULT_PB


def _get_peers_used(ticker: str) -> list[str]:
    """
    Look up which named peer companies actually contributed to this
    ticker's benchmark P/E/P/B (see update_valuation_benchmarks.py —
    this is the subset of FMP's peer list that had valid, non-extreme
    data, not necessarily the full peer group). Returns [] if the
    ticker fell back to Damodaran or the global default — in that
    case there is no specific peer company to name.
    """
    entry = _BENCHMARKS_DATA.get(ticker)
    if entry:
        return entry.get("peers_used", [])
    return []





def valuation_signal(market_data: dict) -> dict | None:
    """
    Compute a valuation signal from yfinance market data.

    Uses a two-tier degradation strategy based on data availability.
    Returns None if no meaningful valuation can be computed, allowing
    the report layer to skip valuation entirely and avoid hallucination.

    Args:
        market_data: dict returned by get_market_data_tool, expected fields:
            - pe_ratio:               float | None  (trailing P/E)
            - price_to_book:          float | None  (P/B ratio)
            - price_to_sales:         float | None  (P/S ratio)
            - ticker:                 str
            - company_name:           str

    Returns:
        dict with keys:
            - valuation_score:   float (-1.0 to +1.0) or None
            - valuation_label:   str or None
            - method:            str — which tier was used
            - reference_only:    bool — True means Tier 2 (P/S), use with caution
            - stale_benchmark:   bool — True if benchmarks are more than 90 days old
            - pe_vs_peers:       str | None — e.g. "36.3 vs peer avg 37.3"
            - detail:            str — plain English explanation of the score
        or None if no data available (Tier 3)
    """

    pe            = market_data.get("pe_ratio")
    pb            = market_data.get("price_to_book")
    ps            = market_data.get("price_to_sales")
    ticker    = market_data.get("ticker") or ""

    # Defensive type check: yfinance has, in practice, returned a
    # non-numeric string (e.g. "Infinity") for these ratio fields when
    # a company's EPS is near zero — a genuine mathematical edge case
    # of the P/E ratio, not a data error (confirmed for BILL,
    # 2026-07-24). Without this check, the comparisons below (pe > 0,
    # etc.) crash with a TypeError instead of gracefully degrading to
    # the next tier.
    if not isinstance(pe, (int, float)):
        pe = None
    if not isinstance(pb, (int, float)):
        pb = None
    if not isinstance(ps, (int, float)):
        ps = None

    sector_pe = _get_benchmark_pe(ticker)
    sector_pb = _get_benchmark_pb(ticker)
    sector_ps = DEFAULT_PS

    def _label(score: float, reference_only: bool) -> str:
        """Map numeric score to human-readable label."""
        if reference_only:
            return "reference only"
        if score > 0.2:
            return "undervalued"
        if score < -0.2:
            return "overvalued"
        return "fairly valued"

    # ────────────────────────────────────────────────────────────────────────
    # TIER 1: P/E + P/B (most reliable)
    # Requires: positive P/E
    # Best for: profitable, well-covered large/mid-cap stocks
    #
    # P/E and P/B are two genuinely independent valuation lenses (earnings-based
    # vs book-value-based) — unlike PEG, which is mathematically derived from
    # P/E itself (PEG = P/E / growth rate) and so double-counts the same
    # information if combined with P/E in a weighted score. Analyst target
    # price ("upside") has also been removed from this signal — it reflects
    # human analyst judgment, not an objective market multiple, and now lives
    # in the separate Consensus Signal so the two remain independently
    # interpretable (agreement between them is meaningful; so is disagreement).
    #
    # If P/B is unavailable, this tier degrades gracefully to P/E alone
    # rather than failing outright — same dynamic-reweighting pattern used
    # in risk_signal.py and quality_signal.py when a sub-component is missing.
    # ────────────────────────────────────────────────────────────────────────
    if pe and pe > 0:

        # P/E score: positive = cheaper than peers, negative = more expensive
        pe_score = (sector_pe - pe) / sector_pe
        pe_score = max(-1.0, min(1.0, pe_score))

        pb_score = None
        if pb and pb > 0:
            pb_score = (sector_pb - pb) / sector_pb
            pb_score = max(-1.0, min(1.0, pb_score))

        if pb_score is not None:
            score  = round(0.6 * pe_score + 0.4 * pb_score, 4)
            pb_detail = f"P/B of {pb} vs peer average {sector_pb} (pb_score={round(pb_score,2)})"
        else:
            score  = round(pe_score, 4)
            pb_detail = "P/B data unavailable — score based on P/E alone"

        label = _label(score, reference_only=False)

        peers_used = _get_peers_used(ticker)
        peers_note = f" (compared against: {', '.join(peers_used)})" if peers_used else ""

        # No named peer group means the benchmark fell back to either
        # Damodaran's industry average or the global S&P 500 default
        # (see _get_benchmark_pe/_get_peers_used) — either way, this is
        # not a company-specific peer comparison, and the report should
        # say so rather than silently presenting it as equivalent to a
        # real peer match (2026-07-27: confirmed ~110-120/250 tickers
        # currently fall into this case, not a rare edge case).
        generic_benchmark_note = (
            "" if peers_used else " Comparison uses a broad S&P 500 average."
        )

        return {
            "valuation_score": score,
            "valuation_label": label,
            "method":          "pe_pb",
            "reference_only":  False,
            "stale_benchmark": BENCHMARKS_STALE,
            "pe_vs_peers": f"{pe} vs peer avg {sector_pe}",
            "peers_used": peers_used,
            "detail": (
                f"P/E of {pe} vs peer average {sector_pe}{peers_note} "
                f"(pe_score={round(pe_score,2)}), "
                f"{pb_detail}. "
                f"Composite: {score} → {label}."
                f"{generic_benchmark_note}"
            ),
        }

    # ────────────────────────────────────────────────────────────────────────
    # TIER 2: P/S only (reference only — loss-making companies)
    # Requires: positive P/S ratio
    # Best for: revenue-generating but unprofitable companies (early-stage)
    # WARNING: P/S ignores profitability — use with caution
    # ────────────────────────────────────────────────────────────────────────
    if ps and ps > 0:

        # P/S score: lower P/S relative to peers = cheaper on revenue basis
        ps_score = (sector_ps - ps) / sector_ps
        ps_score = max(-1.0, min(1.0, ps_score))

        score = round(ps_score, 4)
        label = _label(score, reference_only=True)

        return {
            "valuation_score": score,
            "valuation_label": label,
            "method":          "ps_only",
            "reference_only":  True,
            "stale_benchmark": BENCHMARKS_STALE,
            "pe_vs_peers":    None,
            # No named peer comparison here — this tier uses the
            # global P/S default, not a peer-specific benchmark (see
            # module docstring: Tier 2 is for loss-making companies
            # where P/E-based peer comparison isn't meaningful).
            "peers_used":     [],
            "detail": (
                f"No usable P/E data available for this company. "
                f"P/S of {ps} vs default average {sector_ps} "
                f"(ps_score={round(ps_score,2)}). "
                f"This score is for reference only and does not reflect "
                f"profitability or long-term sustainability."
            ),
        }

    # ────────────────────────────────────────────────────────────────────────
    # TIER 3: No usable data — return None
    # The report layer must handle None gracefully and skip valuation entirely
    # This prevents hallucination when data is unavailable
    # ────────────────────────────────────────────────────────────────────────
    return None