"""
scripts/update_peer_groups.py

Peer Group NAME Updater for EquityMind Layer 2 Quant Engine.

This is the ONLY step in the valuation pipeline that touches FMP's
free-tier quota (250 requests/day) — it fetches, for each ticker in
the universe, which companies FMP considers business-similar peers
(the /stable/stock-peers endpoint). It does nothing else: no P/E, no
P/B, no median computation, no Damodaran fallback.

Why this was split out (2026-07-24) from what used to be a single
update_peer_groups.py that also computed median P/E/P/B: "who are
AAPL's peers" changes on the order of years (a company's business-
similarity peer group is stable unless it undergoes a major strategic
shift) — but "what are those peers' P/E ratios right now" changes
daily with the market. Bundling both into one FMP-bound script meant
the fast-changing part (valuation levels) was held hostage to the
slow-changing part (peer identity) and to FMP's quota. Splitting them
lets update_valuation_benchmarks.py (yfinance-only, no FMP) refresh
valuation levels daily, while this script — the only FMP-quota-bound
step — can run far less often.

Usage:
    python scripts/update_peer_groups.py

Run every few months, or whenever suspecting a company's peer group
should have changed (e.g. a major acquisition/pivot) — NOT on the
daily/weekly cadence used by the rest of Layer 2.

Requires: requests, python-dotenv
"""

import os
import json
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
OUTPUT_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "peer_groups.json"


def _load_target_tickers() -> list[str]:
    universe_path = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"
    with open(universe_path, "r") as f:
        universe = json.load(f)
    return universe["tickers"]


def fetch_peer_symbols(ticker: str) -> list[str]:
    """
    Fetch the list of peer ticker symbols from FMP for one ticker.

    Returns [] if FMP has no peer group for this ticker (or the
    request fails) — this script only reports what FMP actually
    returned. Falling back to Damodaran/global-default when this is
    empty is update_valuation_benchmarks.py's job, not this script's —
    keeping that logic there means this script stays a pure "ask FMP,
    record the answer" step.
    """
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
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Fetching peer group names from FMP...\n")

    peer_groups = {}
    fmp_hit_count = 0

    for i, ticker in enumerate(tickers):
        peers = fetch_peer_symbols(ticker)
        peer_groups[ticker] = peers
        if peers:
            fmp_hit_count += 1

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}")

    output = {
        "updated_at": str(date.today()),
        "source": "FMP stock-peers API",
        "notes": (
            "Peer group IDENTITY only — no valuation numbers here. "
            "Run every few months (peer identity is stable); see "
            "update_valuation_benchmarks.py for the daily-refreshable "
            "P/E-P/B computation that reads this file."
        ),
        "peer_groups": peer_groups,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n✓ peer_groups.json updated — {fmp_hit_count}/{len(tickers)} tickers have an FMP peer group")


if __name__ == "__main__":
    update_peer_groups()
