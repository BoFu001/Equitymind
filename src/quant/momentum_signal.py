"""
src/quant/momentum_signal.py

Momentum Signal Engine — Layer 2 Quantitative Intelligence.

Assesses a stock's price momentum using two independent, academically
validated sub-signals, computed via cross-sectional ranking within a
250-ticker large-cap universe (see scripts/update_momentum_universe.py):

    1. 12-1 Month Momentum (Jegadeesh & Titman, 1993)
       Cumulative return from 12 months ago to 1 month ago (the most
       recent month is excluded to avoid short-term reversal noise).
       Later incorporated into Carhart's (1997) four-factor model.

    2. 52-Week High Position (George & Hwang, 2004)
       Current price's position within the 52-week high-low range.

Both effects have been independently replicated across multiple decades
and international markets (Rouwenhorst 1998; Chui, Titman & Wei 2000 for
momentum; 18/20 major markets for the 52-week high effect) and published
in the Journal of Finance — this is why these two were retained while
RSI, MACD, and simple moving averages were removed: those lack comparable
independent, large-sample academic validation and are more accurately
described as technical-analysis heuristics than validated factors.

IMPORTANT — known limitations (must always be disclosed, not hidden):

    1. GROUP-LEVEL, NOT INDIVIDUAL-STOCK evidence: both effects were
       validated by sorting an entire market into portfolios (e.g. top
       decile vs bottom decile) and comparing GROUP average returns —
       not by predicting any single stock's future price. Applying a
       group-level statistical tendency to one specific stock carries
       real interpretive risk that should not be glossed over.

    2. RECENT PERFORMANCE HAS WEAKENED: 12-1 momentum's out-of-sample
       annualized return over 2014-2024 was approximately 2.23%,
       markedly below both its own historical average (double digits)
       and a simple S&P 500 buy-and-hold over the same period. Momentum
       strategies have also historically suffered severe "crashes"
       during market reversals (e.g. -73% over 3 months in 2009).

    3. UNIVERSE LIMITATION: percentiles are computed within this
       ~250-ticker large-cap universe only (see stock_universe table) —
       not the full US equity market. A ranking here reflects standing
       relative to other large-cap stocks, not the broader market.

    4. NOT A PREDICTION: even where these effects hold, they describe a
       historical statistical tendency, not a guarantee of future
       performance for this specific stock.

Score range: -1.0 (weakest in the universe) to +1.0 (strongest)
Label:       "strong" / "neutral" / "weak"

Academic references:
    Jegadeesh, N., & Titman, S. (1993), 'Returns to Buying Winners and
    Selling Losers: Implications for Stock Market Efficiency',
    Journal of Finance.
    George, T. J., & Hwang, C.-Y. (2004), 'The 52-Week High and
    Momentum Investing', Journal of Finance.
"""

import json
from pathlib import Path
from datetime import date

_MOMENTUM_BENCHMARKS_PATH = Path(__file__).parent / "data" / "momentum_benchmarks.json"


def _load_momentum_benchmarks() -> dict:
    """Load precomputed momentum benchmarks from JSON file."""
    with open(_MOMENTUM_BENCHMARKS_PATH) as f:
        return json.load(f)


_MOMENTUM_BENCHMARKS = _load_momentum_benchmarks()
_BENCHMARKS_DATA = _MOMENTUM_BENCHMARKS.get("benchmarks", {})


def _benchmarks_are_stale() -> bool:
    """Warn if benchmarks are more than 90 days old."""
    updated_at = _MOMENTUM_BENCHMARKS.get("updated_at", "")
    if not updated_at:
        return True
    try:
        updated = date.fromisoformat(updated_at)
        return (date.today() - updated).days > 90
    except ValueError:
        return True


BENCHMARKS_STALE = _benchmarks_are_stale()


def _percentile_to_score(percentile: float) -> float:
    """
    Converts a 0-1 percentile rank to a -1 to +1 score, consistent with
    every other signal engine in this system.
    0.0 (bottom of universe) -> -1.0
    0.5 (median)             ->  0.0
    1.0 (top of universe)    -> +1.0
    """
    return round(percentile * 2 - 1, 4)


def _label(score: float) -> str:
    """Map numeric score to human-readable label."""
    if score > 0.3:
        return "strong"
    if score < -0.3:
        return "weak"
    return "neutral"


def momentum_signal(ticker: str) -> dict | None:
    """
    Compute momentum signals for a ticker from precomputed universe
    benchmarks (see scripts/update_momentum_universe.py).

    Pure function — does not fetch data itself, and unlike
    valuation_signal/consensus_signal, needs nothing from
    get_stock_snapshot() beyond the ticker symbol itself (no live
    price, no market cap, nothing) — it only looks up this ticker's
    precomputed entry in momentum_benchmarks.json. The signature
    reflects this directly (a bare ticker string, not a market_data
    dict) rather than accepting an unused dict for interface
    consistency with the other signal functions — a signature should
    say what a function actually needs, not what would look uniform
    next to its neighbors (2026-07-26).

    Both sub-signals require the ticker to have been successfully
    computed in the batch universe update (at least ~200 trading days
    of price history); if the ticker is missing or was skipped during
    that batch run (e.g. recent IPO), returns None entirely, since
    neither signal can be independently substituted for the other.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys:
            - momentum_12_1_pct:        float | None — raw % return, 12mo excl. last month
            - momentum_12_1_percentile: float | None — 0-1 rank within universe
            - momentum_12_1_score:      float | None — -1 to +1
            - momentum_12_1_label:      str | None — "strong"/"neutral"/"weak"
            - position_52w:             float | None — 0-1, current position in 52w range
            - position_52w_percentile:  float | None — 0-1 rank within universe
            - position_52w_score:       float | None — -1 to +1
            - position_52w_label:       str | None — "strong"/"neutral"/"weak"
            - stale_benchmark:          bool — True if benchmarks are >90 days old
            - detail:                  str — plain English explanation with
                                        mandatory limitation disclosures
        or None if the ticker was not successfully computed in the batch
        universe update (e.g. insufficient price history).
    """
    entry = _BENCHMARKS_DATA.get(ticker)

    if entry is None:
        return None

    momentum_pct        = entry.get("momentum_12_1_pct")
    momentum_percentile = entry.get("momentum_12_1_percentile")
    position            = entry.get("position_52w")
    position_percentile = entry.get("position_52w_percentile")

    if momentum_percentile is None and position_percentile is None:
        return None

    momentum_score = _percentile_to_score(momentum_percentile) if momentum_percentile is not None else None
    momentum_label = _label(momentum_score) if momentum_score is not None else None

    position_score = _percentile_to_score(position_percentile) if position_percentile is not None else None
    position_label = _label(position_score) if position_score is not None else None

    detail_parts = []
    if momentum_pct is not None and momentum_percentile is not None:
        detail_parts.append(
            f"Over the past 12 months (excluding the most recent month), "
            f"this stock returned {momentum_pct}%, ranking at the "
            f"{round(momentum_percentile * 100)}th percentile among the "
            f"~250 large-cap stocks tracked ({momentum_label})."
        )
    if position is not None and position_percentile is not None:
        detail_parts.append(
            f"The stock is currently at {round(position * 100)}% of its "
            f"52-week high-low range, ranking at the "
            f"{round(position_percentile * 100)}th percentile for "
            f"proximity to its 52-week high among the same universe "
            f"({position_label})."
        )

    detail_parts.append(
        "Both figures reflect group-level academic findings (Jegadeesh & "
        "Titman, 1993; George & Hwang, 2004) about how portfolios of "
        "stocks have historically behaved, not a prediction for this "
        "specific stock. Recent (2014-2024) real-world performance of "
        "12-1 momentum has been notably weaker than historical averages, "
        "and momentum strategies have a documented history of severe "
        "reversals ('momentum crashes') following market downturns. "
        "Percentiles are relative to a ~250-ticker large-cap universe, "
        "not the full market."
    )

    return {
        "momentum_12_1_pct":        momentum_pct,
        "momentum_12_1_percentile": momentum_percentile,
        "momentum_12_1_score":      momentum_score,
        "momentum_12_1_label":      momentum_label,
        "position_52w":             position,
        "position_52w_percentile":  position_percentile,
        "position_52w_score":       position_score,
        "position_52w_label":       position_label,
        "stale_benchmark":          BENCHMARKS_STALE,
        "detail": " ".join(detail_parts),
    }
