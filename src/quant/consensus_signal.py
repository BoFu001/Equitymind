"""
src/quant/consensus_signal.py

Consensus Signal Engine — Layer 2 Quantitative Intelligence.

Assesses professional analyst sentiment toward a stock through two
independent sub-signals:

    1. Recommendation Score  — current analyst rating level (1=strong buy
                                to 5=strong sell), converted to [-1, +1]
    2. Upside Score          — analyst mean target price vs current price

The current rating distribution (how many analysts say Buy, Hold, etc.)
is passed through as a raw count alongside them.

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

A rating trend was computed here until 2026-08-24 and has been
removed. yfinance reports only aggregate counts per period, never which
analyst issued which rating when, so a move in the group's weighted
average could not be separated from a change in who was covering the
stock — Apple went from 48 analysts to 44 over three months. The
measure answered two questions at once with no way to tell which one
it was answering.

Score range: -1.0 (bearish consensus) to +1.0 (bullish consensus)
Label:       "bullish" / "neutral" / "bearish"

Academic reference:
    Barber, B., Lehavy, R., McNichols, M., & Trueman, B. (2001),
    'Can Investors Profit from the Prophets? Security Analyst
    Recommendations and Stock Returns', Journal of Finance.
"""

from src.quant.consensus_signal_config import (
    UPSIDE_CAP_PCT,
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


def _sub_label(score: float) -> str:
    """
    Map a single sub-signal's numeric score to a human-readable label.
    Recommendation and upside each get their own, since collapsing them
    into one composite would hide which of the two is driving the
    reading — they answer different questions (where the rating stands
    now, and how far the target price sits above the current one).
    """
    if score > BULLISH_THRESHOLD:
        return "bullish"
    if score < BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


def consensus_signal(consensus_data: dict | None) -> dict | None:
    """
    Compute analyst consensus signals from recommendation and target
    price data.

    Pure function — takes already-fetched data, performs no I/O. Data
    fetching is done separately by consensus_reader.get_consensus_snapshot()
    (for recommendation_mean, target_mean, current_price) and
    consensus_reader.get_rating_counts() (for the current rating
    distribution), called by fetch_data.py, which merges both into the single
    consensus_data dict this function receives (see
    fetch_data._fetch_consensus_inputs()). Both are fetched via
    independent yfinance calls (2026-07-27: no longer shared with
    snapshot_reader.py's get_stock_snapshot() — see consensus_reader.py
    for why), so that a live user question needing only Consensus does
    not have to fetch data other signals would need.

    IMPORTANT — no composite score: recommendation_score and
    upside_score answer two genuinely different questions (where the
    rating stands now / how far the target price sits above the current
    one) and are NOT combined into a single consensus_score. Averaging
    them would produce a number that doesn't correspond to any real
    question a user is asking — a stock rated Hold but carrying a target
    price 40% above its current one would average out to something mild,
    hiding both readings. Each sub-signal is returned independently with
    its own label.

    Degradation strategy: recommendation_score and upside_score require
    only the snapshot portion (recommendation_mean, target_mean,
    current_price) — if these are missing, the whole function returns
    None, since there is no meaningful analyst signal without them.
    latest_rating_counts is allowed to be independently unavailable and
    is simply returned as None.

    Args:
        consensus_data: dict with keys "snapshot" and "rating_counts" (see
            fetch_data._fetch_consensus_inputs()):
            - snapshot: dict from get_consensus_snapshot(), expected fields:
                - recommendation_mean: float | None (1.0-5.0 scale)
                - target_mean:         float | None
                - current_price:       float | None
                - target_high:         float | None
                - target_low:          float | None
                - analyst_count:       int | None
            - rating_counts: dict from get_rating_counts() with keys
                strongBuy/buy/hold/sell/strongSell, or None

    Returns:
        dict with keys:
            - recommendation_score: float (-1.0 to +1.0)
            - recommendation_label: str — "bullish" / "neutral" / "bearish"
            - upside_score:      float (-1.0 to +1.0)
            - upside_pct:        float
            - upside_label:      str — "bullish" / "neutral" / "bearish"
            - low_confidence:    bool — True if analyst_count < MIN_ANALYST_COUNT
            - wide_dispersion:   bool — True if target_high > 3x target_low
            - latest_rating_counts: dict | None — current analyst split
            - detail:            str — plain English explanation of both sub-signals
        or None if consensus_data is None, or recommendation_mean,
        target_mean, or current_price is missing from the snapshot
        (no meaningful analyst signal without them)
    """
    if consensus_data is None:
        return None

    snapshot = consensus_data.get("snapshot") or {}

    recommendation_mean   = snapshot.get("recommendation_mean")
    target_mean           = snapshot.get("target_mean")
    current_price         = snapshot.get("current_price")
    target_high           = snapshot.get("target_high")
    target_low            = snapshot.get("target_low")
    analyst_count         = snapshot.get("analyst_count")

    if recommendation_mean is None or target_mean is None or current_price is None:
        return None

    rec_score = _recommendation_score(recommendation_mean)
    up_score, upside_pct = _upside_score(target_mean, current_price)

    # Straight from the reader: how many analysts currently say Buy vs
    # Sell, which the report layer can show instead of leaning on the
    # single averaged recommendation_mean.
    latest_rating_counts = consensus_data.get("rating_counts")

    recommendation_label = _sub_label(rec_score)
    upside_label = _sub_label(up_score)

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
        "Analyst ratings carry a well-documented systematic optimism bias — "
        "'sell' ratings are rare in practice, so a positive score here "
        "should be read as 'positive within a system that skews positive,' "
        "not as a neutral, unbiased signal.",
    ]
    # low_confidence / wide_dispersion warnings are NOT duplicated into
    # detail here -- formatters.format_consensus() already surfaces both
    # as standalone Note lines using the returned booleans below.
    # detail's job is disclosing the methodology bias, not restating
    # these two flags.


    return {
        # Raw inputs, included alongside the computed results so the
        # report layer (and ultimately the user) can see what each
        # judgment is actually based on — e.g. "bullish (score=0.65)"
        # means little without knowing whether that came from 3
        # analysts or 30, or what the actual recommendation_mean was.
        "recommendation_mean":  recommendation_mean,
        "target_mean":          target_mean,
        "current_price":        current_price,
        "target_high":          target_high,
        "target_low":           target_low,
        "analyst_count":        analyst_count,
        "latest_rating_counts": latest_rating_counts,

        "recommendation_score": round(rec_score, 4),
        "recommendation_label": recommendation_label,
        "upside_score":         round(up_score, 4),
        "upside_pct":           upside_pct,
        "upside_label":         upside_label,
        "low_confidence":       low_confidence,
        "wide_dispersion":      wide_dispersion,
        "detail": " ".join(detail_parts),
    }
