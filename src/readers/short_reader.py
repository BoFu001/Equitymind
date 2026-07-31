"""
src/readers/short_reader.py

Fetches raw short-interest data needed for Short Signal calculations
(Short Interest level, Days to Cover, month-over-month change) —
independent of snapshot_reader.py and every other reader, following
the same one-reader-per-signal pattern (see risk_reader.py,
consensus_reader.py).

IMPORTANT — this is a slow-updating data source, not a live one.
Verified 2026-07-30 against real AAPL data: dateShortInterest and
sharesShortPreviousMonthDate were exactly one month apart (2026-06-15
to 2026-07-15), consistent with FINRA/NASDAQ's official short-interest
reporting cadence. Every date field below MUST be surfaced to the
formatter layer and displayed alongside the numbers it dates — a
Short Interest percentage or Days to Cover figure with no reporting
date attached is not meaningfully interpretable (a number needs a
"when", the same way a raw score needs a benchmark to compare
against — see valuation_signal.py's peer-comparison requirement).

Academic grounding (verified via literature search, 2026-07-31):
    - Short Interest level: well-established predictor of future
      returns — Asquith, Pathak & Ritter (2005, Journal of Financial
      Economics); Boehmer, Jones & Zhang (2008); replicated
      internationally (Review of Asset Pricing Studies). High short
      interest predicts lower future returns; safe to present as a
      primary signal.
    - Days to Cover (shortRatio): no direct academic
      return-predictability literature found — this is a
      liquidity/squeeze-risk metric (how crowded a short position is
      relative to trading volume), not a directional signal. Must not
      be presented with the same "predicts returns" framing as Short
      Interest level.
    - Month-over-month Short Interest change: MIXED evidence — one
      paper (Surprise in short interest, ScienceDirect 2023) finds "no
      marginal predictive power" beyond the level; another (Short
      interest, stock returns and credit ratings, ScienceDirect 2019)
      finds a significant effect but ONLY for low-credit-rating firms.
      Not a standalone signal — display as descriptive context only,
      with an explicit caveat, never with the same confidence as the
      level itself.
"""

import yfinance as yf


def get_short_inputs(ticker: str) -> dict | None:
    """
    Fetches raw short-interest data for Short Signal calculations.

    Returns None only if the .info call itself fails. Unlike
    risk_reader.get_risk_inputs() (where stock_prices is a single
    shared foundation all four sub-signals depend on), Short Signal's
    fields are independent of each other — short_percent_of_float and
    short_ratio/shares_short are not derived from one another, so no
    single field is treated as mandatory here. Every field below is
    allowed to be individually None; callers (short_signal.py) decide
    which sub-signals can still be computed from whatever is present.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys:
            - short_percent_of_float: float, e.g. 0.01 for 1% (share of
                                       float currently sold short), or
                                       None if unavailable
            - short_ratio:            float, Days to Cover — total
                                       shares short divided by average
                                       daily volume, or None if
                                       unavailable
            - shares_short:           int, current shares sold short,
                                       or None if unavailable
            - shares_short_prior_month: int, shares sold short as of
                                       the previous reporting date, or
                                       None if unavailable
            - date_short_interest:    int, Unix timestamp for the
                                       CURRENT shares_short figure —
                                       MUST be shown alongside any
                                       number derived from shares_short
                                       or short_percent_of_float
            - date_short_interest_prior_month: int, Unix timestamp for
                                       shares_short_prior_month — MUST
                                       be shown alongside any
                                       month-over-month change figure
            - float_shares:           int, shares available for public
                                       trading (the denominator behind
                                       short_percent_of_float) — also
                                       useful for flagging unusually
                                       concentrated ownership structures
                                       (e.g. float_shares far smaller
                                       than total shares outstanding)
        or None if the .info call itself failed.
    """
    try:
        info = yf.Ticker(ticker).info

        return {
            "short_percent_of_float":          info.get("shortPercentOfFloat"),
            "short_ratio":                     info.get("shortRatio"),
            "shares_short":                    info.get("sharesShort"),
            "shares_short_prior_month":        info.get("sharesShortPriorMonth"),
            "date_short_interest":             info.get("dateShortInterest"),
            "date_short_interest_prior_month": info.get("sharesShortPreviousMonthDate"),
            "float_shares":                    info.get("floatShares"),
        }

    except Exception as e:
        print(f"  [get_short_inputs] Error fetching {ticker}: {e}")
        return None
