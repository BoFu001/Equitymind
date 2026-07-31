"""
src/quant/short_signal.py

Short Signal Engine — Layer 2 Quantitative Intelligence.

Reports three independent short-selling metrics, each on its own terms
(no combined score — see below):
    1. Short Interest    — share of float currently sold short
    2. Days to Cover      — short position size relative to trading
                            volume (liquidity/squeeze-risk, NOT a
                            directional predictor)
    3. Month-over-month change — descriptive context only

This is a pure function — it takes already-fetched data as input and
performs no I/O itself. Data fetching lives in
short_reader.get_short_inputs(), called separately by fetch_data.py,
keeping this file testable without mocking any network calls (same
pattern as risk_signal.py and valuation_signal.py).

IMPORTANT — evidence tiering (reviewed 2026-07-31; see
short_signal_config.py for the full account of what was removed and
why): only Short Interest level has well-established academic support
(Asquith, Pathak & Ritter 2005; Boehmer, Jones & Zhang 2008) — reported
as a raw percentage with no derived label or score, since no sourced
threshold was found for classifying it into tiers. Days to Cover has a
sourced industry-practice "elevated" threshold (not peer-reviewed) —
labelled, but explicitly framed as a liquidity/squeeze-risk indicator,
never as bearish/bullish. Month-over-month change has MIXED academic
evidence and is reported as a raw percentage only, with no label.

IMPORTANT — this is a slow-updating data source, not a live one. The
underlying FINRA/NASDAQ short-interest reporting cadence is
approximately monthly (verified 2026-07-30: dateShortInterest and its
prior-month counterpart were exactly one month apart for a real AAPL
query). Every number this function returns MUST be displayed alongside
its reporting date (date_short_interest) — a percentage or ratio with
no "as of" date attached is not meaningfully interpretable (see
short_reader.py module docstring).

Academic references:
    Asquith, P., Pathak, P., & Ritter, J. (2005), 'Short Interest,
    Institutional Ownership, and Stock Returns', Journal of Financial
    Economics.
    Boehmer, E., Jones, C., & Zhang, X. (2008), 'Which Shorts Are
    Informed?', Journal of Finance.
"""

from src.quant.short_signal_config import DAYS_TO_COVER_ELEVATED_THRESHOLD


def _days_to_cover_label(short_ratio: float) -> str:
    """Map a Days to Cover value to a human-readable label."""
    if short_ratio >= DAYS_TO_COVER_ELEVATED_THRESHOLD:
        return "elevated"
    return "normal"


def short_signal(short_inputs: dict | None) -> dict | None:
    """
    Compute Short Signal from short_reader.get_short_inputs().

    Degradation strategy: the three sub-signals are independent of
    each other's data (see short_reader.py) — each is computed if its
    own required field(s) are present. Returns None only if ALL THREE
    would be None — i.e. short_inputs itself is None, or every
    relevant field within it is None.

    Args:
        short_inputs: dict from get_short_inputs(), expected fields:
            - short_percent_of_float: float | None
            - short_ratio:            float | None
            - shares_short:           int | None
            - shares_short_prior_month: int | None
            - date_short_interest:    int | None (Unix timestamp)
            - date_short_interest_prior_month: int | None (Unix timestamp)
            - float_shares:           int | None

    Returns:
        dict with keys:
            - short_interest_pct:     float | None — raw share of float,
                                       no derived label or score (see
                                       module docstring for why)
            - days_to_cover:          float | None — raw Days to Cover
            - days_to_cover_label:    str | None — "elevated"/"normal"
                                       (liquidity/squeeze-risk reading,
                                       NOT a directional score)
            - mom_change_pct:         float | None — raw percentage,
                                       no derived label
            - date_short_interest:            int | None — Unix timestamp,
                                       MUST be shown alongside
                                       short_interest_pct/days_to_cover
            - date_short_interest_prior_month: int | None — Unix
                                       timestamp, MUST be shown alongside
                                       mom_change_pct
            - float_shares:           int | None
        or None if short_inputs is None or every relevant field is None.
    """
    if short_inputs is None:
        return None

    short_percent       = short_inputs.get("short_percent_of_float")
    short_ratio         = short_inputs.get("short_ratio")
    shares_short        = short_inputs.get("shares_short")
    shares_short_prior  = short_inputs.get("shares_short_prior_month")
    date_current        = short_inputs.get("date_short_interest")
    date_prior          = short_inputs.get("date_short_interest_prior_month")
    float_shares        = short_inputs.get("float_shares")

    days_to_cover_label = None
    if short_ratio is not None:
        days_to_cover_label = _days_to_cover_label(short_ratio)

    mom_change_pct = None
    if shares_short is not None and shares_short_prior is not None and shares_short_prior != 0:
        mom_change_pct = round(
            (shares_short - shares_short_prior) / shares_short_prior * 100, 2
        )

    if short_percent is None and short_ratio is None and mom_change_pct is None:
        return None

    return {
        "short_interest_pct":               short_percent,
        "days_to_cover":                    short_ratio,
        "days_to_cover_label":              days_to_cover_label,
        "mom_change_pct":                   mom_change_pct,
        "date_short_interest":              date_current,
        "date_short_interest_prior_month":  date_prior,
        "float_shares":                     float_shares,
    }
