"""
scripts/currency_check.py

Shared currency filter for EquityMind's Layer 2 batch scripts —
build_stock_universe.py, update_quant_signals.py, and
update_financial_history.py all need to answer the same question
("is this ticker's financial reporting in USD?") and must agree on
the answer, so the check lives in one place rather than being
duplicated three times.

Background: yfinance's financialCurrency field (from stock.info)
reflects the currency a company's underlying financial statements are
reported in — this can differ from "currency" (the currency the stock
itself trades in on the exchange), which is USD for any US-listed
ADR regardless of the issuer's home country. A company like Toyota
(TM) trades in USD but reports financials in JPY — its net_income,
total_revenue etc. from .financials/.cashflow/.balance_sheet are raw
JPY figures, not USD, and would silently corrupt any cross-company
dollar comparison if stored without this check.

financialCurrency == None is treated as PASS, not a rejection — this
is metadata that is sometimes simply missing (confirmed for FISV,
which reports in USD with normal financial data; None here is
evidence of incomplete metadata, not evidence of foreign currency).
Only an explicit non-USD currency code is treated as a real signal to
exclude a ticker.
"""

import yfinance as yf


def is_usd_reporter(ticker: str) -> bool:
    """
    Returns True if this ticker's financial statements are reported in
    USD (or the currency metadata is missing — treated as pass, not a
    rejection). Returns False only when yfinance explicitly reports a
    non-USD financialCurrency (e.g. JPY, TWD, EUR).

    Fails open (returns True) on any lookup error — a network hiccup
    here shouldn't silently exclude a legitimate US company; batch
    scripts downstream already have their own retry/skip handling for
    tickers that fail for other reasons.
    """
    try:
        info = yf.Ticker(ticker).info
        currency = info.get("financialCurrency")
        return currency is None or currency == "USD"
    except Exception:
        return True
