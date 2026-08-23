"""
src/readers/consensus_reader.py

Fetches ALL data Consensus Signal Engine needs — the point-in-time
analyst fields and the current rating distribution — independently of
snapshot_reader.py — even though this means recommendation_mean,
target_mean, current_price, target_high, target_low, and
analyst_count are fetched via a SEPARATE yfinance .info call here,
duplicating part of what snapshot_reader.get_stock_snapshot() also
fetches, rather than sharing one call.

This is a deliberate reversal (2026-07-27) of this file's original
design, which shared those six fields with snapshot_reader.py to
avoid a duplicate API call. That tradeoff made sense for the batch
script (update_quant_signals.py: always computing all 5 signals for
250 tickers, so sharing one call was pure savings) but not for live
user questions, where a question like "what do analysts think of
Apple?" only needs Consensus — forcing a full snapshot fetch just to
get six fields nobody else in that request needs is the wrong
tradeoff there. Full per-signal independence (this file owning
everything Consensus needs, snapshot_reader.py owning only what's
shared elsewhere) is a prerequisite for a future "which data does
this question need" routing node to skip data sources cleanly, one
signal at a time — see project notes, 2026-07-27.

Split out from the former market_data.py on 2026-07-27 (alongside
snapshot_reader.py and risk_reader.py).
"""

import yfinance as yf


def get_consensus_snapshot(ticker: str) -> dict | None:
    """
    Fetches the point-in-time analyst fields Consensus Signal Engine
    needs — a separate yfinance .info call from
    snapshot_reader.get_stock_snapshot(), even though both pull from
    the same underlying yfinance data (see module docstring for why).

    Returns None if the .info call itself fails — callers should not
    assume a partial dict is possible here (unlike get_stock_snapshot,
    which returns whatever fields it manages to get).

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys: recommendation_mean, target_mean, current_price,
        target_high, target_low, analyst_count
        or None if the fetch failed.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "recommendation_mean": info.get("recommendationMean"),
            "target_mean":         info.get("targetMeanPrice"),
            "current_price":       info.get("currentPrice"),
            "target_high":         info.get("targetHighPrice"),
            "target_low":          info.get("targetLowPrice"),
            "analyst_count":       info.get("numberOfAnalystOpinions"),
        }
    except Exception as e:
        print(f"  [get_consensus_snapshot] Error fetching {ticker}: {e}")
        return None


def get_rating_counts(ticker: str) -> dict | None:
    """
    Fetches the current period's analyst rating distribution — how many
    analysts say Strong Buy, Buy, Hold, Sell, Strong Sell right now.

    This is a raw count, not a derived figure: it is the one piece of
    analyst data here that needs no aggregation we cannot explain.
    yfinance's .recommendations returns several months of history; only
    the most recent period is read. The earlier periods fed a rating
    trend until 2026-08-24, when it was removed — a shift in the group's
    average rating cannot be separated from a shift in which analysts
    are covering the stock (Apple went from 48 analysts to 44 over three
    months), so the trend was measuring two things at once with no way
    to tell them apart.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys strongBuy, buy, hold, sell, strongSell,
        or None if no rating data is available.
    """
    try:
        recommendations = yf.Ticker(ticker).recommendations

        if recommendations is None or len(recommendations) == 0:
            return None

        row = recommendations.iloc[0]   # most recent period
        return {
            "strongBuy":  int(row["strongBuy"]),
            "buy":        int(row["buy"]),
            "hold":       int(row["hold"]),
            "sell":       int(row["sell"]),
            "strongSell": int(row["strongSell"]),
        }

    except Exception as e:
        print(f"  [get_rating_counts] Error fetching {ticker}: {e}")
        return None
