"""
src/quant/momentum_signal.py

Momentum Signal Engine — Layer 2 Quantitative Intelligence.

Assesses a stock's price momentum using two independent, academically
validated sub-signals, computed via cross-sectional ranking within a
250-ticker large-cap universe (see scripts/update_momentum_benchmarks.py):

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
       not by predicting any single stock's future price.

    2. RECENT PERFORMANCE HAS WEAKENED: 12-1 momentum's out-of-sample
       annualized return over 2014-2024 was approximately 2.23%,
       markedly below both its own historical average (double digits)
       and a simple S&P 500 buy-and-hold over the same period.

    3. UNIVERSE LIMITATION: percentiles are computed within this
       ~250-ticker large-cap universe only — not the full US equity
       market.

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


def momentum_signal(momentum_inputs: dict | None) -> dict | None:
    """
    Compute momentum signals from momentum_reader.get_momentum_inputs().

    Pure function — does no fetching itself. Both sub-signals require
    the ticker to have been successfully computed in the batch
    universe update (at least ~200 trading days of price history); if
    momentum_inputs is None or has no percentile data, returns None.

    Args:
        momentum_inputs: dict from get_momentum_inputs(), or None

    Returns:
        dict with keys:
            - momentum_12_1_pct:        float | None
            - momentum_12_1_percentile: float | None
            - momentum_12_1_score:      float | None — -1 to +1
            - momentum_12_1_label:      str | None
            - position_52w:             float | None
            - position_52w_percentile:  float | None
            - position_52w_score:       float | None — -1 to +1
            - position_52w_label:       str | None
            - detail:                  str
        or None if momentum_inputs is unavailable.
    """
    if momentum_inputs is None:
        return None

    momentum_pct        = momentum_inputs.get("momentum_12_1_pct")
    momentum_percentile = momentum_inputs.get("momentum_12_1_percentile")
    position            = momentum_inputs.get("position_52w")
    position_percentile = momentum_inputs.get("position_52w_percentile")

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
        "detail": " ".join(detail_parts),
    }
