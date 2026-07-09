"""
src/quant/consensus_signal.py

Consensus Signal Engine — Layer 2 Quantitative Intelligence.

Assesses professional analyst sentiment toward a stock, combining three
independent sub-signals:

    1. Recommendation Score  — current analyst rating level (1=strong buy
                                to 5=strong sell), converted to [-1, +1]
    2. Upside Score          — analyst mean target price vs current price
    3. Trend Score           — how the weighted analyst rating has moved
                                over the observed history window

Unlike Valuation/Momentum/Risk/Quality, this signal reflects HUMAN
JUDGMENT, not an objective market calculation — it must be presented
with that distinction made clear.

Known limitation (disclosed, not corrected): sell-side analyst ratings
carry a well-documented systematic optimism bias (Barber, Lehavy,
McNichols & Trueman, 2001) — "sell" ratings are rare in practice, partly
because analysts' firms often seek investment banking business from the
same companies they rate, and because the career risk of a wrong "sell"
call is asymmetrically worse than a wrong "buy" call. A high
recommendation score should be read as "positive within a system that
skews positive," not as a neutral, unbiased signal.

Trend Score methodology note: this is a WEIGHTED-AVERAGE-OF-THE-GROUP
comparison (current period vs the oldest available period), not a
true analyst-by-analyst revision tracker of the kind used by
institutional data providers (e.g. I/B/E/S — see Barber et al. and
similar literature on "recommendation revisions" defined per-analyst).
The free yfinance data used here only reports aggregate counts per
period, not which individual analyst issued which rating when — so this
trend can be influenced by analysts starting/stopping coverage, not
only by existing analysts changing their mind. It is a reasonable,
transparent proxy for group-level sentiment direction, but should not be
presented as equivalent to institutional-grade analyst revision tracking.

Score range: -1.0 (bearish consensus) to +1.0 (bullish consensus)
Label:       "bullish" / "neutral" / "bearish"

Academic reference:
    Barber, B., Lehavy, R., McNichols, M., & Trueman, B. (2001),
    'Can Investors Profit from the Prophets? Security Analyst
    Recommendations and Stock Returns', Journal of Finance.
"""

from src.quant.consensus_signal_config import (
    UPSIDE_CAP_PCT,
    TREND_CAP_POINTS,
    MIN_ANALYST_COUNT,
    WIDE_TARGET_RANGE_RATIO,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
)


def _recommendation_score(recommendation_mean: float) -> float:
    """
    Sub-signal 1: current analyst rating level.
    recommendation_mean is on a 1 (strong buy) to 5 (strong sell) scale.
    Maps to [-1, +1] with 3 (hold) as the neutral midpoint.
    """
    score = (3 - recommendation_mean) / 2
    return max(-1.0, min(1.0, score))


def _upside_score(target_mean: float, current_price: float) -> tuple[float, float]:
    """
    Sub-signal 2: analyst mean target price vs current price.
    Returns (upside_score, upside_pct).
    """
    upside_pct = (target_mean - current_price) / current_price
    score = upside_pct / UPSIDE_CAP_PCT
    return max(-1.0, min(1.0, score)), round(upside_pct * 100, 2)


def _weighted_rating(period: dict) -> float | None:
    """
    Computes the analyst-count-weighted average rating for a single period,
    on the same 1 (strong buy) to 5 (strong sell) scale as recommendation_mean.
    Returns None if the period has zero contributing analysts.
    """
    total = (period["strongBuy"] + period["buy"] + period["hold"] +
             period["sell"] + period["strongSell"])
    if total == 0:
        return None
    weighted_sum = (1 * period["strongBuy"] + 2 * period["buy"] +
                     3 * period["hold"] + 4 * period["sell"] +
                     5 * period["strongSell"])
    return weighted_sum / total


def _trend_score(consensus_inputs: dict) -> tuple[float | None, str]:
    """
    Sub-signal 3: direction of change in the weighted analyst rating,
    comparing the most recent period to the oldest available period.

    Uses PROPORTIONS (weighted averages), not raw analyst counts, because
    the total number of contributing analysts can fluctuate slightly
    period to period (individual analysts starting/stopping coverage) —
    comparing raw counts would introduce noise unrelated to actual
    sentiment change.

    Returns (trend_score, detail) or (None, reason) if a trend cannot be
    computed (fewer than 2 usable periods).
    """
    periods = consensus_inputs.get("periods", [])
    if len(periods) < 2:
        return None, "Insufficient rating history to compute a trend (need at least 2 periods)."

    current_weighted = _weighted_rating(periods[0])    # most recent (e.g. "0m")
    oldest_weighted   = _weighted_rating(periods[-1])  # oldest available

    if current_weighted is None or oldest_weighted is None:
        return None, "Could not compute a weighted rating for the available periods (zero analysts)."

    # Negative raw_trend means the weighted rating fell (moved toward "buy")
    # i.e. sentiment improved. We negate so positive trend_score = improving.
    raw_trend = current_weighted - oldest_weighted
    trend_score = max(-1.0, min(1.0, -raw_trend / TREND_CAP_POINTS))

    direction = (
        "improving" if trend_score > 0.05 else
        "deteriorating" if trend_score < -0.05 else
        "stable"
    )
    oldest_period_label = periods[-1]["period"]
    detail = (
        f"Analyst sentiment {direction} over the observed window "
        f"(weighted rating {oldest_weighted:.2f} -> {current_weighted:.2f}, "
        f"from {oldest_period_label} to present). This reflects the "
        f"group's aggregate rating distribution, not individual analyst "
        f"revisions — it can be influenced by analysts starting or "
        f"stopping coverage, not only by existing analysts changing "
        f"their view."
    )
    return round(trend_score, 4), detail


def _sub_label(score: float) -> str:
    """
    Map a single sub-signal's numeric score to a human-readable label.
    Used independently for recommendation, upside, and trend — each
    sub-signal gets its own label, since combining them into one
    composite score/label would obscure which dimension is driving
    the reading (see module docstring: these three answer different
    questions — current standing, future price target, and recent
    directional change — and should not be collapsed into one number).
    """
    if score > BULLISH_THRESHOLD:
        return "bullish"
    if score < BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


def consensus_signal(market_data: dict, consensus_inputs: dict | None) -> dict | None:
    """
    Compute analyst consensus signals from recommendation and target
    price data.

    Pure function — takes already-fetched data, performs no I/O. Data
    fetching is done separately by market_data.get_stock_snapshot() (for
    recommendation_mean, target_mean, current_price) and
    market_data.get_consensus_inputs() (for historical rating trend),
    called by quant_engine.py before this function runs.

    IMPORTANT — no composite score: recommendation_score, upside_score,
    and trend_score answer three genuinely different questions (current
    standing / future price target / recent directional change) and are
    NOT combined into a single consensus_score. Averaging them would
    produce a number that doesn't correspond to any real question a user
    is asking — e.g. a stock with a strongly bullish current rating but a
    deteriorating recent trend would average out to "neutral," hiding
    both the strong current standing AND the concerning trend. Each
    sub-signal is returned independently with its own label, so the
    report layer (and ultimately the user) can see the full picture
    rather than a blended, uninterpretable average.

    Degradation strategy: recommendation_score and upside_score require
    only the current snapshot (recommendation_mean, target_mean,
    current_price) — if these are missing, the whole function returns
    None, since there is no meaningful analyst signal without them.
    trend_score is allowed to be independently unavailable (e.g.
    insufficient rating history) — it is simply returned as None,
    with no downstream reweighting needed since there is no composite
    to reweight.

    Args:
        market_data: dict from get_stock_snapshot(), expected fields:
            - recommendation_mean: float | None (1.0-5.0 scale)
            - target_mean:         float | None
            - current_price:       float | None
            - target_high:         float | None
            - target_low:          float | None
            - analyst_count:       int | None
        consensus_inputs: dict from get_consensus_inputs(), or None

    Returns:
        dict with keys:
            - recommendation_score: float (-1.0 to +1.0)
            - recommendation_label: str — "bullish" / "neutral" / "bearish"
            - upside_score:      float (-1.0 to +1.0)
            - upside_pct:        float
            - upside_label:      str — "bullish" / "neutral" / "bearish"
            - trend_score:       float | None
            - trend_label:       str | None — "improving" / "stable" / "deteriorating"
            - low_confidence:    bool — True if analyst_count < MIN_ANALYST_COUNT
            - wide_dispersion:   bool — True if target_high > 3x target_low
            - detail:            str — plain English explanation of all sub-signals
        or None if recommendation_mean, target_mean, or current_price
        is missing (no meaningful analyst signal without them)
    """
    recommendation_mean   = market_data.get("recommendation_mean")
    target_mean           = market_data.get("target_mean")
    current_price         = market_data.get("current_price")
    target_high           = market_data.get("target_high")
    target_low            = market_data.get("target_low")
    analyst_count         = market_data.get("analyst_count")

    if recommendation_mean is None or target_mean is None or current_price is None:
        return None

    rec_score = _recommendation_score(recommendation_mean)
    up_score, upside_pct = _upside_score(target_mean, current_price)

    trend_score, trend_detail = (None, "No rating history data available.")
    if consensus_inputs is not None:
        trend_score, trend_detail = _trend_score(consensus_inputs)

    recommendation_label = _sub_label(rec_score)
    upside_label = _sub_label(up_score)
    trend_label = (
        None if trend_score is None else
        ("improving" if trend_score > 0.05 else
         "deteriorating" if trend_score < -0.05 else
         "stable")
    )

    low_confidence = analyst_count is not None and analyst_count < MIN_ANALYST_COUNT
    wide_dispersion = (
        target_high is not None and target_low is not None and
        target_low > 0 and target_high > WIDE_TARGET_RANGE_RATIO * target_low
    )

    detail_parts = [
        f"Recommendation mean {recommendation_mean} "
        f"(recommendation_score={round(rec_score,2)}, {recommendation_label}), "
        f"analyst target upside {upside_pct}% "
        f"(upside_score={round(up_score,2)}, {upside_label}).",
        trend_detail,
        "Analyst ratings carry a well-documented systematic optimism bias — "
        "'sell' ratings are rare in practice, so a positive score here "
        "should be read as 'positive within a system that skews positive,' "
        "not as a neutral, unbiased signal.",
    ]
    if low_confidence:
        detail_parts.append(
            f"⚠️ Only {analyst_count} analysts cover this stock — small "
            f"sample size, lower confidence."
        )
    if wide_dispersion:
        detail_parts.append(
            f"⚠️ Wide target price dispersion (${target_low} to ${target_high}) "
            f"indicates significant disagreement among analysts about this "
            f"stock's future value."
        )

    return {
        "recommendation_score": round(rec_score, 4),
        "recommendation_label": recommendation_label,
        "upside_score":         round(up_score, 4),
        "upside_pct":           upside_pct,
        "upside_label":         upside_label,
        "trend_score":          trend_score,
        "trend_label":          trend_label,
        "low_confidence":       low_confidence,
        "wide_dispersion":      wide_dispersion,
        "detail": " ".join(detail_parts),
    }
