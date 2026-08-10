"""
scripts/currency_check.py

Shared check: is this ticker's financial reporting in USD? A stock
can trade in USD (currency) while reporting financials in another
currency (financialCurrency) — e.g. Toyota (TM), USD-traded but
JPY-reporting. Comparing raw financial figures across tickers requires
financialCurrency == USD, not just USD trading.
"""

import yfinance as yf


def is_usd_reporter(ticker: str) -> bool:
    """True only if financialCurrency is confirmed as USD. Anything
    unverifiable (lookup failure, missing profile, missing or non-USD
    currency) returns False."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return False

    if not info.get("symbol"):
        return False

    return info.get("financialCurrency") == "USD"
