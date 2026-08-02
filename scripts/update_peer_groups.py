"""
scripts/update_peer_groups.py

Fetches each ticker's FMP business-similar peers, filters out non-USD
reporters, writes the result to stock_universe.peers. The only step
in the valuation pipeline that uses FMP's quota (250 requests/day).

Usage:
    python scripts/update_peer_groups.py
"""

import os
import sys
import json
import psycopg2
import requests
from pathlib import Path
from dotenv import load_dotenv
from psycopg2.extras import Json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.currency_check import is_usd_reporter

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def _load_target_tickers(cursor) -> list[str]:
    cursor.execute("SELECT ticker FROM stock_universe")
    return [row[0] for row in cursor.fetchall()]


def fetch_peer_symbols(ticker: str) -> list[str]:
    """Fetches peer tickers from FMP. Returns [] on no data or failure."""
    url = f"https://financialmodelingprep.com/stable/stock-peers?symbol={ticker}&apikey={FMP_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        peers = json.loads(r.text)
        return [p.get("symbol") for p in peers if p.get("symbol") and p.get("symbol") != ticker]
    except Exception:
        return []


def update_peer_groups():
    print("EquityMind — Peer Group Name Updater")
    print("=" * 50)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    tickers = _load_target_tickers(cursor)
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Fetching peer group names from FMP...\n")

    fmp_hit_count = 0

    for i, ticker in enumerate(tickers):
        raw_peers = fetch_peer_symbols(ticker)
        peers = [p for p in raw_peers if is_usd_reporter(p)]
        excluded = [p for p in raw_peers if p not in peers]
        if excluded:
            print(f"  [{ticker}] Excluded non-USD peers: {excluded}")
        cursor.execute(
            "UPDATE stock_universe SET peers = %s WHERE ticker = %s",
            (Json(peers), ticker),
        )
        if peers:
            fmp_hit_count += 1

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}")
            conn.commit()

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n✓ stock_universe table updated — {fmp_hit_count}/{len(tickers)} tickers have an FMP peer group")


if __name__ == "__main__":
    update_peer_groups()
