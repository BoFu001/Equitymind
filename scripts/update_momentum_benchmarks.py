"""
scripts/update_momentum_benchmarks.py

Momentum Benchmark Updater for EquityMind Layer 2 Quant Engine.

Computes two independent, academically-validated momentum sub-signals for
every ticker in stock_universe.json, using yfinance only (no FMP — this
script does not touch the FMP daily quota at all):

    1. 12-1 Month Momentum (Jegadeesh & Titman, 1993)
       Cumulative return from 12 months ago to 1 month ago (excludes the
       most recent month to avoid short-term reversal noise). This is the
       standard academic momentum factor, later incorporated into
       Carhart's (1997) four-factor model.

    2. 52-Week High Position (George & Hwang, 2004)
       Current price's position within the 52-week high-low range
       (0 = at the 52-week low, 1 = at the 52-week high).

Both signals were originally validated via CROSS-SECTIONAL RANKING (sort
all stocks, compare top decile vs bottom decile), not as standalone
per-stock thresholds — so this script computes the raw value for every
stock in the universe, then converts each to a PERCENTILE RANK within
that universe. This is why a batch script (not a per-request calculation)
is required: ranking needs the full universe computed together.

IMPORTANT — known limitations (must be disclosed to users, not hidden):
    - Both effects are validated at the PORTFOLIO/GROUP level (buy top
      decile, sell bottom decile), not as predictions for any single
      stock's future performance.
    - Recent (2014-2024) out-of-sample performance of 12-1 momentum has
      been notably weaker than prior decades (annualized ~2.23% vs prior
      double-digit returns), and momentum crashes (e.g. 2009: -73% in 3
      months) are a documented, recurring risk following market reversals.
    - This universe (250 hand-picked large-cap tickers) is NOT the full
      market — rankings reflect relative position within this specific
      large-cap universe only, not the broader US equity market.

Usage:
    python scripts/update_momentum_benchmarks.py

Run whenever stock_universe.json is refreshed, or periodically (e.g.
monthly) — unlike peer_benchmarks.json, this has no FMP quota constraint,
so it can be run more frequently if desired.

Requires: yfinance, numpy (already in the project's environment)
"""

import json
from datetime import date
from pathlib import Path

import yfinance as yf

OUTPUT_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "momentum_benchmarks.json"

# ~1 month and ~12 months in trading days (US market convention)
TRADING_DAYS_PER_MONTH = 22
TRADING_DAYS_PER_YEAR  = 252

# Minimum trading days required to compute both signals reliably.
# Below this, a stock (e.g. very recent IPO) is skipped rather than
# reporting an unreliable estimate.
MIN_TRADING_DAYS = 200


def _load_target_tickers() -> list[str]:
    universe_path = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"
    with open(universe_path, "r") as f:
        universe = json.load(f)
    return universe["tickers"]


def compute_momentum_raw(ticker: str) -> dict | None:
    """
    Fetches ~1 year of price history and computes the two raw momentum
    values (not yet ranked) for a single ticker.

    Returns None if price history is unavailable or too short (e.g.
    very recent IPO) — this stock is simply excluded from the ranking
    rather than given an unreliable estimate.
    """
    try:
        hist = yf.Ticker(ticker).history(period="1y")
    except Exception as e:
        print(f"    Skipping {ticker}: {e}")
        return None

    if hist is None or len(hist) < MIN_TRADING_DAYS:
        return None

    closes = hist["Close"].dropna()  # some data sources have occasional
                                       # NaN values on specific days (seen
                                       # in practice, e.g. ORCL's earliest
                                       # trading day) — drop them before
                                       # indexing so a missing single-day
                                       # value doesn't silently produce NaN
                                       # results downstream.
    if len(closes) < MIN_TRADING_DAYS:
        return None

    current_price  = closes.iloc[-1]
    price_1m_ago   = closes.iloc[-TRADING_DAYS_PER_MONTH]
    price_12m_ago  = closes.iloc[0]  # earliest available point (~12 months back)
    high_52w = closes.max()
    low_52w  = closes.min()

    import math
    if price_12m_ago == 0 or math.isnan(price_12m_ago) or math.isnan(price_1m_ago):
        momentum_12_1_pct = None
    else:
        momentum_12_1_pct = (price_1m_ago - price_12m_ago) / price_12m_ago

    if high_52w == low_52w or math.isnan(high_52w) or math.isnan(low_52w) or math.isnan(current_price):
        position_52w = None
    else:
        position_52w = (current_price - low_52w) / (high_52w - low_52w)

    return {
        "momentum_12_1_pct": momentum_12_1_pct,
        "position_52w":      position_52w,
    }


def compute_percentile_ranks(raw_data: dict) -> tuple[dict, dict]:
    """
    Converts raw values into percentile ranks within the universe.
    0.0 = lowest in the universe, 1.0 = highest in the universe —
    intuitive direction: higher percentile = stronger momentum /
    closer to 52-week high.

    Returns (momentum_percentiles, position_percentiles), each a dict
    of {ticker: percentile}.
    """
    momentum_pairs = [
        (t, v["momentum_12_1_pct"]) for t, v in raw_data.items()
        if v is not None and v["momentum_12_1_pct"] is not None
    ]
    position_pairs = [
        (t, v["position_52w"]) for t, v in raw_data.items()
        if v is not None and v["position_52w"] is not None
    ]

    momentum_sorted = sorted(momentum_pairs, key=lambda x: x[1])
    position_sorted = sorted(position_pairs, key=lambda x: x[1])

    n_momentum = len(momentum_sorted)
    n_position = len(position_sorted)

    momentum_percentiles = {
        t: round(i / (n_momentum - 1), 4) if n_momentum > 1 else 0.5
        for i, (t, _) in enumerate(momentum_sorted)
    }
    position_percentiles = {
        t: round(i / (n_position - 1), 4) if n_position > 1 else 0.5
        for i, (t, _) in enumerate(position_sorted)
    }

    return momentum_percentiles, position_percentiles


def update_momentum_benchmarks():
    print("EquityMind — Momentum Benchmark Updater")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Fetching price history and computing raw momentum values...\n")

    raw_data = {}
    for i, ticker in enumerate(tickers):
        result = compute_momentum_raw(ticker)
        raw_data[ticker] = result
        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}")

    computed_count = sum(1 for v in raw_data.values() if v is not None)
    print(f"\nSuccessfully computed raw values for {computed_count}/{len(tickers)} tickers")

    print("Computing percentile ranks across the universe...")
    momentum_percentiles, position_percentiles = compute_percentile_ranks(raw_data)

    benchmarks = {}
    for ticker in tickers:
        raw = raw_data.get(ticker)
        if raw is None:
            benchmarks[ticker] = None
            continue
        benchmarks[ticker] = {
            "momentum_12_1_pct":        round(raw["momentum_12_1_pct"] * 100, 2) if raw["momentum_12_1_pct"] is not None else None,
            "momentum_12_1_percentile": momentum_percentiles.get(ticker),
            "position_52w":             round(raw["position_52w"], 4) if raw["position_52w"] is not None else None,
            "position_52w_percentile":  position_percentiles.get(ticker),
        }

    output = {
        "updated_at": str(date.today()),
        "source": "yfinance price history (1y) — no FMP dependency",
        "universe_size": len(tickers),
        "computed_count": computed_count,
        "notes": (
            "Percentiles are computed within this 250-ticker large-cap "
            "universe only, not the broader market. Both signals reflect "
            "portfolio/group-level academic findings (Jegadeesh & Titman "
            "1993; George & Hwang 2004), not per-stock predictions. Recent "
            "(2014-2024) real-world performance of 12-1 momentum has been "
            "notably weaker than historical averages."
        ),
        "benchmarks": benchmarks,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n✓ momentum_benchmarks.json updated — {computed_count}/{len(tickers)} tickers computed")


if __name__ == "__main__":
    update_momentum_benchmarks()
