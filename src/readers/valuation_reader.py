"""
src/readers/valuation_reader.py

Fetches the point-in-time ratios valuation_signal() needs — the
target ticker's own P/E and P/S, plus the same ratios for each of
its peers (from stock_universe.peers), all via yfinance.
"""

import os
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def _get_peers(ticker: str) -> list[str]:
    """Looks up this ticker's peers from stock_universe.peers."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT peers FROM stock_universe WHERE ticker = %s", (ticker,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row[0] if row and row[0] else []
    except Exception:
        return []


def _get_ratios(symbol: str) -> dict | None:
    """Fetches trailingPE/priceToSalesTrailing12Months for one ticker via yfinance."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "pe": info.get("trailingPE"),
            "ps": info.get("priceToSalesTrailing12Months"),
        }
    except Exception:
        return None


def get_valuation_inputs(ticker: str) -> dict | None:
    """
    Fetches the target ticker's own P/E and P/S, plus its peers'
    identity and each peer's P/E and P/S.

    Returns:
        dict with keys:
            ticker, pe_ratio, price_to_sales,
            peers (list of peer symbols),
            peer_ratios (dict: symbol -> {"pe": ..., "ps": ...})
        or None if the target ticker's own fetch failed.
    """
    try:
        info = yf.Ticker(ticker).info
        own = {
            "ticker":         ticker,
            "pe_ratio":       info.get("trailingPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
        }
    except Exception as e:
        print(f"  [get_valuation_inputs] Error fetching {ticker}: {e}")
        return None

    peers = _get_peers(ticker)
    peer_ratios = {}
    for symbol in peers:
        if symbol == ticker:
            continue
        ratios = _get_ratios(symbol)
        if ratios:
            peer_ratios[symbol] = ratios

    own["peers"] = peers
    own["peer_ratios"] = peer_ratios
    return own
