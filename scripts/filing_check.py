"""
scripts/filing_check.py

Shared checks on how a company reports to the SEC. Two independent
questions, each serving a different requirement:

is_usd_reporter — are the financials denominated in USD? Comparing raw
figures across tickers (F-Score, valuation multiples) requires
financialCurrency == USD, not just USD trading. A stock can trade in
USD while reporting in another currency — e.g. Toyota (TM), USD-traded
but JPY-reporting.

files_20f_only — is this a foreign private issuer? Those file 20-F
rather than 10-K, so no Item 1 business description is available for
sector overview embeddings.
"""

import time

import yfinance as yf
from edgar import Company, set_identity

set_identity("Bo Fu bofu001@gmail.com")

# SEC allows 10 requests/second. 0.5s is well under that, and throttling
# matters here because a rate-limited request raises, and the exception
# path returns False — silently admitting a foreign issuer.
SEC_THROTTLE_SECONDS = 0.5


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


def files_20f_only(ticker: str) -> bool:
    """True if the most recent annual report is a 20-F, meaning no Item 1
    business description is available. Cannot be inferred from yfinance:
    GFS reports in USD and is tagged country=United States, yet files
    20-F. Filing status can change — NXPI, RCL, SHOP and TEAM all have
    both forms in their history after losing foreign private issuer
    status — so only the newest filing is checked, not historical counts.
    Returns False on any failure, and False for companies with no annual
    report yet (a recent IPO is not a foreign issuer)."""
    time.sleep(SEC_THROTTLE_SECONDS)
    try:
        # form= matches amendments (10-K/A, 20-F/A) by prefix; those are
        # filed later than the original and would distort "most recent".
        annuals = [
            f for f in Company(ticker).get_filings(form=["10-K", "20-F"])
            if f.form in ("10-K", "20-F")
        ]
    except Exception:
        return False

    if not annuals:
        return False
    else:
        return max(annuals, key=lambda f: f.filing_date).form == "20-F"
