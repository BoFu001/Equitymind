"""
src/quant/consensus_signal.py

Consensus Signal Engine — Layer 2 Quantitative Intelligence.

Assesses professional analyst sentiment toward a stock through two
independent sub-signals:

    1. Recommendation mean   — the analyst rating level on its own
                                1-5 scale, where 1 is strong buy
    2. Upside                — how far the analyst mean target sits
                                above the current price, as a percentage

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
call is asymmetrically worse than a wrong "buy" call. A rating near
the buy end should be read as "positive within a system that skews
positive," not as a neutral, unbiased signal.

A rating trend was computed here until 2026-08-24 and has been
removed. yfinance reports only aggregate counts per period, never which
analyst issued which rating when, so a move in the group's weighted
average could not be separated from a change in who was covering the
stock — Apple went from 48 analysts to 44 over three months. The
measure answered two questions at once with no way to tell which one
it was answering.

Nothing here is transformed. Both figures are reported as the data
source gives them: the 1-5 rating mean (1 is best, so it sorts
ascending) and the implied move as a percentage. A +/-1 rescaling and
a bullish/neutral/bearish label were dropped on 2026-08-24 — neither
carried information the raw figures and the rating distribution did
not already carry more plainly.

Academic reference:
    Barber, B., Lehavy, R., McNichols, M., & Trueman, B. (2001),
    'Can Investors Profit from the Prophets? Security Analyst
    Recommendations and Stock Returns', Journal of Finance.
"""

from src.quant.consensus_signal_config import (
    MIN_ANALYST_COUNT,
    WIDE_TARGET_RANGE_RATIO,
)


def _upside_pct(target_mean: float, current_price: float) -> float:
    """
    Sub-signal 2: how far the analyst mean target sits above the current
    price, as a percentage.

    This used to be divided by a 50% cap and clipped to produce a score
    on the same +/-1 scale as everything else, which gave every target
    above 50% the same value — five companies shared +1.0 on 2026-08-23,
    and MSTR at 92% ranked behind ORCL at 68% on nothing but database
    row order. The percentage needs no scale of its own: it is already
    continuous, already sorts correctly, and already means something to
    a reader without being explained.
    """
    return round((target_mean - current_price) / current_price * 100, 2)


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

    IMPORTANT — no composite score: recommendation_mean and
    upside_pct answer two genuinely different questions (where the
    rating stands now / how far the target price sits above the current
    one) and are NOT combined into a single consensus_score. Averaging
    them would produce a number that doesn't correspond to any real
    question a user is asking — a stock rated Hold but carrying a target
    price 40% above its current one would average out to something mild,
    hiding both readings. Each is returned on its own.

    Degradation strategy: recommendation_mean and upside_pct require
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
            - recommendation_mean: float — the 1-5 analyst mean, 1 best
            - upside_pct:        float — implied move, as a percentage
            - low_confidence:    bool — True if analyst_count < MIN_ANALYST_COUNT
            - wide_dispersion:   bool — True if target_high > 3x target_low
            - latest_rating_counts: dict | None — current analyst split
            - detail:            str — the optimism-bias disclosure
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

    upside_pct = _upside_pct(target_mean, current_price)

    # Straight from the reader: how many analysts currently say Buy vs
    # Sell, which the report layer can show instead of leaning on the
    # single averaged recommendation_mean.
    latest_rating_counts = consensus_data.get("rating_counts")


    low_confidence = analyst_count is not None and analyst_count < MIN_ANALYST_COUNT
    wide_dispersion = (
        target_high is not None and target_low is not None and
        target_low > 0 and target_high > WIDE_TARGET_RANGE_RATIO * target_low
    )

    # The numbers themselves are not repeated here — format_consensus()
    # already prints the recommendation mean, the target range and the
    # rating split, and low_confidence / wide_dispersion go out as their
    # own Note lines. All this string has to do is disclose the bias
    # that the numbers cannot show on their own.
    detail = (
        "Analyst ratings carry a well-documented systematic optimism bias — "
        "'sell' ratings are rare in practice, so a positive reading here "
        "should be taken as 'positive within a system that skews positive,' "
        "not as a neutral, unbiased signal."
    )


    return {
        # Raw inputs, included alongside the computed results so the
        # report layer (and ultimately the user) can see what each
        # judgment is actually based on — a recommendation mean of
        # 2.2 means little without knowing whether it came from 3
        # analysts or 30.
        "recommendation_mean":  recommendation_mean,
        "target_mean":          target_mean,
        "current_price":        current_price,
        "target_high":          target_high,
        "target_low":           target_low,
        "analyst_count":        analyst_count,
        "latest_rating_counts": latest_rating_counts,

        "upside_pct":           upside_pct,
        "low_confidence":       low_confidence,
        "wide_dispersion":      wide_dispersion,
        "detail":               detail,
    }
