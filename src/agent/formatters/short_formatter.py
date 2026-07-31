"""
src/agent/formatters/short_formatter.py

Formats short_signal() results for the LLM — see
src/quant/short_signal.py for the data this consumes.

Three independent sub-signals, NOT combined into a single score (same
principle as risk_formatter.py) — but unlike Risk (where all four
sub-signals share equal academic standing), Short Signal's three
components carry DIFFERENT evidentiary weight (see short_signal.py's
module docstring), and this formatter surfaces that distinction
explicitly rather than presenting all three with equal confidence.

Every reporting date is shown alongside the number it dates — a
percentage or ratio with no "as of" date attached is not meaningfully
interpretable (see short_reader.py module docstring: this is a
slow-updating, ~monthly data source, not a live one).
"""

from datetime import datetime


def _format_date(unix_ts) -> str:
    """Converts a Unix timestamp to a plain-English date, or a
    placeholder if the timestamp is missing."""
    if unix_ts is None:
        return "date unavailable"
    try:
        return datetime.fromtimestamp(unix_ts).strftime("%B %d, %Y")
    except (ValueError, OSError):
        return "date unavailable"


def format_short(short: dict | None) -> str:
    """
    Shows raw inputs (percentage of float, days to cover, raw share
    counts) alongside the computed labels — not just the labels alone
    (same principle already applied to risk_formatter.py /
    valuation_formatter.py: a label like "elevated" means little
    without the actual number and its reporting date).
    """
    if not short:
        return "  Short Interest: Insufficient data.\n"

    text = "  Short Interest:\n"

    if short.get("short_interest_pct") is not None:
        date_str = _format_date(short.get("date_short_interest"))
        text += (
            f"    Short Interest: {round(short['short_interest_pct'] * 100, 2)}% "
            f"of float, as of {date_str}.\n"
        )
    else:
        text += "    Short Interest: Unavailable.\n"

    if short.get("days_to_cover") is not None:
        text += (
            f"    Days to Cover: {short['days_to_cover']} days "
            f"({short['days_to_cover_label']}) — liquidity/squeeze-risk "
            f"indicator, not a directional predictor (industry-practice "
            f"threshold, not peer-reviewed).\n"
        )
    else:
        text += "    Days to Cover: Unavailable.\n"

    if short.get("mom_change_pct") is not None:
        prior_date_str = _format_date(short.get("date_short_interest_prior_month"))
        direction = "increased" if short["mom_change_pct"] > 0 else "decreased"
        text += (
            f"    Month-over-month change: shares sold short {direction} "
            f"{abs(short['mom_change_pct'])}% since {prior_date_str} — "
            f"descriptive context only, not an independently scored "
            f"signal (academic evidence on this specific metric is "
            f"mixed and inconclusive).\n"
        )

    if short.get("float_shares") is not None:
        text += f"    Float shares (denominator): {short['float_shares']:,}\n"


    return text
