"""
scripts/update_momentum_benchmarks.py

Momentum Benchmark Updater for EquityMind Layer 2 Quant Engine.

Computes two independent, academically-validated momentum sub-signals for
every ticker in the stock_universe table, using yfinance only (no FMP — this
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

Run whenever the stock_universe table is refreshed, or periodically (e.g.
monthly) — unlike peer_benchmarks.json, this has no FMP quota constraint,
so it can be run more frequently if desired.

Requires: yfinance, numpy (already in the project's environment)
"""


import sys
from pathlib import Path

import psycopg2
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL


# ~1 month and ~12 months in trading days (US market convention)
TRADING_DAYS_PER_MONTH = 22
TRADING_DAYS_PER_YEAR  = 252

# Minimum trading days required to compute both signals reliably.
# Below this, a stock (e.g. very recent IPO) is skipped rather than
# reporting an unreliable estimate.
MIN_TRADING_DAYS = 200


def _load_target_tickers() -> list[str]:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM stock_universe")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tickers


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
        print(f"    Skipping {ticker}: {e}", flush=True)
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


INSERT_SQL = """
    INSERT INTO momentum_benchmarks (
        ticker, momentum_12_1_pct, momentum_12_1_percentile,
        position_52w, position_52w_percentile, updated_at
    )
    VALUES (%s, %s, %s, %s, %s, NOW())
    ON CONFLICT (ticker) DO UPDATE SET
        momentum_12_1_pct = EXCLUDED.momentum_12_1_pct,
        momentum_12_1_percentile = EXCLUDED.momentum_12_1_percentile,
        position_52w = EXCLUDED.position_52w,
        position_52w_percentile = EXCLUDED.position_52w_percentile,
        updated_at = NOW();
"""


def update_momentum_benchmarks():
    print("EquityMind — Momentum Benchmark Updater", flush=True)
    print("=" * 50, flush=True)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers", flush=True)
    print("Fetching price history and computing raw momentum values...\n", flush=True)

    raw_data = {}
    for i, ticker in enumerate(tickers):
        result = compute_momentum_raw(ticker)
        raw_data[ticker] = result
        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}", flush=True)

    computed_count = sum(1 for v in raw_data.values() if v is not None)
    print(f"\nSuccessfully computed raw values for {computed_count}/{len(tickers)} tickers", flush=True)

    print("Computing percentile ranks across the universe...", flush=True)
    momentum_percentiles, position_percentiles = compute_percentile_ranks(raw_data)

    def _to_float(value):
        # numpy.float64 passed straight to psycopg2 gets rendered as
        # "np.float64(0.5)" in the SQL text, which Postgres tries to
        # parse as a schema-qualified identifier ("np"."float64(...)"),
        # failing every write — same issue documented in
        # update_quant_signals.py's _to_float().
        return None if value is None else float(value)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM momentum_benchmarks WHERE ticker != ALL(%s)", (tickers,))
    conn.commit()

    write_success = 0
    print("Writing to momentum_benchmarks table...", flush=True)
    for i, ticker in enumerate(tickers):
        raw = raw_data.get(ticker)
        if raw is None:
            continue
        row = (
            ticker,
            _to_float(round(raw["momentum_12_1_pct"] * 100, 2)) if raw["momentum_12_1_pct"] is not None else None,
            _to_float(momentum_percentiles.get(ticker)),
            _to_float(round(raw["position_52w"], 4)) if raw["position_52w"] is not None else None,
            _to_float(position_percentiles.get(ticker)),
        )
        try:
            cursor.execute(INSERT_SQL, row)
            conn.commit()
            write_success += 1
        except Exception as e:
            print(f"    DB write failed for {ticker}: {e}", flush=True)
            conn.rollback()

        if (i + 1) % 50 == 0:
            print(f"  ...written {i + 1}/{len(tickers)}", flush=True)

    cursor.close()
    conn.close()

    print(f"\n✓ momentum_benchmarks table updated — {write_success}/{len(tickers)} tickers written", flush=True)


if __name__ == "__main__":
    update_momentum_benchmarks()
