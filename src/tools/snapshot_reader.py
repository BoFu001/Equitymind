"""
src/tools/snapshot_reader.py

Fetches a single point-in-time snapshot of company fundamentals for
a ticker — one yfinance .info call, shared by multiple consumers
(not owned by any single signal):

    - Displayed directly in reports (format_stock_snapshot): price,
      market cap, sector/industry, EPS, dividend yield.
    - valuation_signal() reads pe_ratio, price_to_book, price_to_sales.
    - consensus_signal() no longer reads anything from this file as
      of 2026-07-27 — all fields it needs (recommendation_mean,
      target_mean, current_price, target_high, target_low,
      analyst_count) moved to
      consensus_reader.get_consensus_snapshot(), fetched independently
      via a separate yfinance call rather than shared with this file
      (see consensus_reader.py for why).

Split out from the former market_data.py on 2026-07-27 (alongside
risk_reader.py and consensus_reader.py) — one reader file per data
point, named for what it provides rather than grouped under a generic
"market_data" umbrella, so a future "which data does this question
need" routing node can address each data source independently rather
than treating market_data as an indivisible unit.
"""

import yfinance as yf


def get_stock_snapshot(ticker: str) -> dict | None:
    """
    Fetches fundamentals and technical indicators for a ticker.
    Returns a dict with price, P/E, revenue, RSI, MACD, SMA or None if failed.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # Basic fundamentals
        snapshot = {
            "ticker":             ticker,
            "company_name":       info.get("longName", ticker),
            "current_price":      info.get("currentPrice"),
            "market_cap":         info.get("marketCap"),
            "pe_ratio":           info.get("trailingPE"),
            "forward_pe":         info.get("forwardPE"),
            "revenue":            info.get("totalRevenue"),
            "profit_margin":      info.get("profitMargins"),
            "52w_high":           info.get("fiftyTwoWeekHigh"),
            "52w_low":            info.get("fiftyTwoWeekLow"),
            "sector":             info.get("sector"),
            "industry":           info.get("industry"),
            # Earnings and dividends
            "eps_trailing":       info.get("trailingEps"),
            "eps_forward":        info.get("forwardEps"),
            "dividend_yield":     info.get("dividendYield"),
            # Valuation ratios for Layer 2 quant engine
            "price_to_sales":     info.get("priceToSalesTrailing12Months"),
            "price_to_book":      info.get("priceToBook"),
        }

        # NOTE: RSI/MACD/SMA technical indicators were deliberately removed —
        # they lack independent, large-sample academic validation (see
        # momentum_signal.py docstring). Momentum is now measured via
        # 12-1 month momentum and 52-week high position, computed from
        # precomputed universe benchmarks, not from this function.

        return snapshot

    except Exception as e:
        print(f"  [get_stock_snapshot] Error fetching {ticker}: {e}")
        return None
