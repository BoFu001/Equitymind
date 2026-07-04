"""
src/quant/valuation_signal.py

Valuation Signal Engine — Layer 2 Quantitative Intelligence.

Assesses whether a stock is overvalued or undervalued using a three-tier
degradation strategy based on data availability:

    Tier 1 (most reliable):  P/E + PEG + analyst upside  → method = "pe_peg_upside"
    Tier 2 (reliable):       P/E + analyst upside         → method = "pe_upside"
    Tier 3 (reference only): P/S ratio (loss-making firms) → method = "ps_only"
    Tier 4 (no data):        Returns None — caller must handle gracefully

Score range: -1.0 (severely overvalued) to +1.0 (severely undervalued)
Label:       "overvalued" / "fairly valued" / "undervalued" / "reference only" / None

Academic references:
    - P/E relative valuation: Damodaran (2012), Investment Valuation
    - PEG ratio: Lynch (1989), One Up on Wall Street
    - P/S for loss-making firms: Fisher (1984), Super Stocks
"""


import json
from pathlib import Path
from datetime import date

# ─────────────────────────────────────────────
# Load sector benchmarks from JSON
# Updated quarterly by scripts/update_benchmarks.py
# ─────────────────────────────────────────────
_BENCHMARKS_PATH = Path(__file__).parent / "data" / "sector_benchmarks.json"

def _load_benchmarks() -> dict:
    """Load sector benchmarks from JSON file."""
    with open(_BENCHMARKS_PATH) as f:
        return json.load(f)

_BENCHMARKS = _load_benchmarks()

SECTOR_PE = _BENCHMARKS.get("sector_pe", {})
SECTOR_PS = _BENCHMARKS.get("sector_ps", {})
DEFAULT_PE = _BENCHMARKS.get("default_pe", 20)
DEFAULT_PS = _BENCHMARKS.get("default_ps", 2)

# ── Staleness check ──────────────────────────
# Warn if benchmarks are more than 90 days old
def _benchmarks_are_stale() -> bool:
    updated_at = _BENCHMARKS.get("updated_at", "")
    if not updated_at:
        return True
    try:
        updated = date.fromisoformat(updated_at)
        return (date.today() - updated).days > 90
    except ValueError:
        return True

BENCHMARKS_STALE = _benchmarks_are_stale()





def valuation_signal(market_data: dict) -> dict | None:
    """
    Compute a valuation signal from yfinance market data.

    Uses a three-tier degradation strategy based on data availability.
    Returns None if no meaningful valuation can be computed, allowing
    the report layer to skip valuation entirely and avoid hallucination.

    Args:
        market_data: dict returned by get_market_data_tool, expected fields:
            - pe_ratio:               float | None  (trailing P/E)
            - peg_ratio:              float | None  (PEG ratio)
            - price_to_sales:         float | None  (P/S ratio)
            - sector:                 str
            - target_mean:            float | None  (analyst mean target price)
            - current_price:          float | None
            - company_name:           str

    Returns:
        dict with keys:
            - valuation_score:   float (-1.0 to +1.0) or None
            - valuation_label:   str or None
            - method:            str — which tier was used
            - reference_only:    bool — True means Tier 3 (P/S), use with caution
            - stale_benchmark:   bool — True if benchmarks are more than 90 days old
            - upside_pct:        float | None — % upside to analyst target
            - pe_vs_sector:      str | None — e.g. "36.3 vs sector avg 28"
            - detail:            str — plain English explanation of the score
        or None if no data available (Tier 4)
    """

    pe            = market_data.get("pe_ratio")
    peg           = market_data.get("peg_ratio")
    ps            = market_data.get("price_to_sales")
    sector        = market_data.get("sector") or ""
    target_mean   = market_data.get("target_mean")
    current_price = market_data.get("current_price")

    sector_pe = SECTOR_PE.get(sector, DEFAULT_PE)
    sector_ps = SECTOR_PS.get(sector, DEFAULT_PS)

    # ── Shared helper: compute analyst upside score ──────────────────────────
    def _upside_score_and_pct():
        """Returns (upside_score, upside_pct) or (0.0, None) if no target price."""
        if target_mean and current_price and target_mean > 0 and current_price > 0:
            pct   = (target_mean - current_price) / current_price
            score = max(-1.0, min(1.0, pct * 2))  # 50% upside → +1.0
            return score, round(pct * 100, 2)
        return 0.0, None

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
    # TIER 1: P/E + PEG + analyst upside (most reliable)
    # Requires: positive P/E, positive PEG, any price data
    # Best for: profitable, well-covered large/mid-cap stocks
    # ────────────────────────────────────────────────────────────────────────
    earnings_growth = market_data.get("earnings_growth")
    peg_valid = (
        peg and peg > 0 and peg < 20  # PEG > 20 is meaningless
        and (earnings_growth is None or earnings_growth > 0)  # skip if negative growth
    )
    if pe and pe > 0 and peg_valid:

        # P/E score: positive = cheaper than sector, negative = more expensive
        pe_score = (sector_pe - pe) / sector_pe
        pe_score = max(-1.0, min(1.0, pe_score))

        # PEG score: PEG < 1 is undervalued, PEG > 1 is overvalued
        # Normalised so that PEG = 1 → 0, PEG = 0 → +1, PEG = 2 → -1
        peg_score = max(-1.0, min(1.0, (1.0 - peg) / 1.0))

        upside_score, upside_pct = _upside_score_and_pct()

        # Weighted composite: PEG carries most weight as it accounts for growth
        score = round(
            0.35 * pe_score +
            0.40 * peg_score +
            0.25 * upside_score,
            4
        )

        label = _label(score, reference_only=False)

        return {
            "valuation_score": score,
            "valuation_label": label,
            "method":          "pe_peg_upside",
            "reference_only":  False,
            "stale_benchmark": BENCHMARKS_STALE,
            "upside_pct":      upside_pct,
            "pe_vs_sector":    f"{pe} vs sector avg {sector_pe}",
            "detail": (
                f"P/E of {pe} vs sector average {sector_pe} "
                f"(pe_score={round(pe_score,2)}), "
                f"PEG of {round(peg,2)} "
                f"(peg_score={round(peg_score,2)}), "
                f"analyst upside {upside_pct}% "
                f"(upside_score={round(upside_score,2)}). "
                f"Composite: {score} → {label}."
            ),
        }

    # ────────────────────────────────────────────────────────────────────────
    # TIER 2: P/E + analyst upside (reliable, no PEG available)
    # Requires: positive P/E
    # Best for: profitable companies without reliable growth estimates
    # ────────────────────────────────────────────────────────────────────────
    if pe and pe > 0:

        pe_score = (sector_pe - pe) / sector_pe
        pe_score = max(-1.0, min(1.0, pe_score))

        upside_score, upside_pct = _upside_score_and_pct()

        score = round(0.6 * pe_score + 0.4 * upside_score, 4)
        label = _label(score, reference_only=False)

        return {
            "valuation_score": score,
            "valuation_label": label,
            "method":          "pe_upside",
            "reference_only":  False,
            "stale_benchmark": BENCHMARKS_STALE,
            "upside_pct":      upside_pct,
            "pe_vs_sector":    f"{pe} vs sector avg {sector_pe}",
            "detail": (
                f"{'PEG excluded — negative earnings growth' if earnings_growth is not None and earnings_growth < 0 else 'PEG not available'}. "
                f"P/E of {pe} vs sector average {sector_pe} "
                f"(pe_score={round(pe_score,2)}), "
                f"analyst upside {upside_pct}% "
                f"(upside_score={round(upside_score,2)}). "
                f"Composite: {score} → {label}."
            ),
        }

    # ────────────────────────────────────────────────────────────────────────
    # TIER 3: P/S only (reference only — loss-making companies)
    # Requires: positive P/S ratio
    # Best for: revenue-generating but unprofitable companies (early-stage)
    # WARNING: P/S ignores profitability — use with caution
    # ────────────────────────────────────────────────────────────────────────
    if ps and ps > 0:

        # P/S score: lower P/S relative to sector = cheaper on revenue basis
        ps_score = (sector_ps - ps) / sector_ps
        ps_score = max(-1.0, min(1.0, ps_score))

        upside_score, upside_pct = _upside_score_and_pct()

        score = round(0.6 * ps_score + 0.4 * upside_score, 4)
        label = _label(score, reference_only=True)

        return {
            "valuation_score": score,
            "valuation_label": label,
            "method":          "ps_only",
            "reference_only":  True,
            "stale_benchmark": BENCHMARKS_STALE,
            "upside_pct":      upside_pct,
            "pe_vs_sector":    None,
            "detail": (
                f"Company is loss-making (no positive P/E). "
                f"P/S of {ps} vs sector average {sector_ps} "
                f"(ps_score={round(ps_score,2)}). "
                f"This score is for reference only and does not reflect "
                f"profitability or long-term sustainability."
            ),
        }

    # ────────────────────────────────────────────────────────────────────────
    # TIER 4: No usable data — return None
    # The report layer must handle None gracefully and skip valuation entirely
    # This prevents hallucination when data is unavailable
    # ────────────────────────────────────────────────────────────────────────
    return None