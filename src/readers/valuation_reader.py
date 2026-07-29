"""
src/tools/valuation_reader.py

Fetches the point-in-time ratios valuation_signal() needs — a
separate yfinance .info call from snapshot_reader.py's
get_stock_snapshot(), even though both pull from the same underlying
yfinance data.

Split out from snapshot_reader.py on 2026-07-27, following the same
principle already applied to consensus_reader.py: every signal's data
point should be independently fetchable, so a future "which data does
this question need" routing node can skip data sources cleanly, one
signal at a time, without needing to reason about which signals share
which underlying calls. This means pe_ratio/price_to_book/
price_to_sales are now fetched via a dedicated call here, duplicating
part of what snapshot_reader.get_stock_snapshot() also fetches, rather
than sharing one call — a deliberate tradeoff (see
consensus_reader.py's docstring for the same reasoning applied there).
"""

import yfinance as yf


def get_valuation_inputs(ticker: str) -> dict | None:
    """
    Fetches the ratios valuation_signal() needs.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys: ticker, pe_ratio, price_to_book, price_to_sales
        or None if the fetch failed.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":         ticker,
            "pe_ratio":       info.get("trailingPE"),
            "price_to_book":  info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
        }
    except Exception as e:
        print(f"  [get_valuation_inputs] Error fetching {ticker}: {e}")
        return None
