"""
scripts/update_peer_groups.py

Fetches each ticker's FMP business-similar peers, filters out non-USD
reporters, writes the result to stock_universe.peers. The only step
in the valuation pipeline that uses FMP's quota (250 requests/day).

Usage:
    python scripts/update_peer_groups.py
"""

import sys
import json
import psycopg2
import requests
from pathlib import Path
from psycopg2.extras import Json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.filing_check import is_usd_reporter
from config import FMP_API_KEY, DATABASE_URL


def _load_target_tickers(cursor) -> list[str]:
    # Only tickers missing peers — a company's business-similarity peer
    # group doesn't change day to day, and FMP's quota (250 requests/day)
    # is shared with the whole universe, so re-fetching all 250 existing
    # tickers every day could starve newly-added tickers of quota before
    # they're even reached. Only new (or previously-failed, still-NULL)
    # tickers need processing.
    cursor.execute("SELECT ticker FROM stock_universe WHERE peers IS NULL")
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
    print("EquityMind — Peer Group Name Updater", flush=True)
    print("=" * 50, flush=True)
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    tickers = _load_target_tickers(cursor)
    print(f"\nTickers needing peers: {len(tickers)}", flush=True)
    if tickers:
        print(f"  {tickers}", flush=True)
    print("Fetching peer group names from FMP...\n", flush=True)
    fmp_hit_count = 0
    still_empty_count = 0
    for i, ticker in enumerate(tickers):
        raw_peers = fetch_peer_symbols(ticker)
        peers = [p for p in raw_peers if is_usd_reporter(p)]
        excluded = [p for p in raw_peers if p not in peers]
        if excluded:
            print(f"  [{ticker}] Excluded non-USD peers: {excluded}", flush=True)

        if not peers:
            # Empty result — whether from the FMP call failing, a
            # genuine no-peers response, or every candidate getting
            # filtered out for non-USD reporting — is deliberately
            # left NULL rather than written as []. Writing [] would
            # make this ticker look "already processed" and permanently
            # exempt it from the WHERE peers IS NULL retry logic in
            # _load_target_tickers(), so a transient FMP failure (like
            # HAL's on 2026-08-14, confirmed by a manual re-fetch minutes
            # later returning 8 real peers) would never self-correct.
            # Leaving it NULL means every future run retries it for
            # free — including tickers with no real peers (e.g. SPCX,
            # a pre-IPO company), which costs a wasted FMP call each day
            # but means the day FMP does add real peer data, it gets
            # picked up automatically with no manual intervention.
            print(f"  [{ticker}] no peers after filtering — leaving NULL for retry next run", flush=True)
            still_empty_count += 1
        else:
            print(f"  [{ticker}] peers: {peers}", flush=True)
            cursor.execute(
                "UPDATE stock_universe SET peers = %s, updated_at = NOW() WHERE ticker = %s",
                (Json(peers), ticker),
            )
            fmp_hit_count += 1

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}", flush=True)
            conn.commit()
    conn.commit()
    cursor.close()
    conn.close()
    print(f"\n✓ stock_universe table updated — {fmp_hit_count}/{len(tickers)} new tickers now have an FMP peer group", flush=True)
    if still_empty_count:
        print(f"  {still_empty_count} ticker(s) left NULL (empty result) — will retry next run", flush=True)


if __name__ == "__main__":
    update_peer_groups()
