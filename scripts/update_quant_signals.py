"""
scripts/update_quant_signals.py

Quant Signal Updater for EquityMind Layer 2 — computes all 5 cacheable
signals (Valuation, Momentum, Risk, Quality, Consensus — News Sentiment
is deliberately excluded, see quant_signals table docstring) for every
ticker in stock_universe.json and stores them in the quant_signals
table (see init_db_quant_signals.py).

Each signal's COMPLETE return dict is stored verbatim as JSONB — the
exact same dict shape that quant_engine.py produces when computing
live, so format_valuation()/format_risk()/etc. work identically on a
cached row or a fresh computation. A few scalar score columns are also
populated for discovery_suggest's screening queries — see
init_db_quant_signals.py for which ones and why. Scalar column names
match their source JSONB key exactly (e.g. position_52w_score, not
momentum_52w_score) to avoid any ambiguity about which field they were
copied from.

Signals that return None (e.g. quality_signal() with <2 fiscal years
of data, risk_signal() with <60 trading days of price history,
momentum_signal() for a ticker missing from momentum_benchmarks.json)
are stored as NULL — this is a legitimate, expected outcome, not an
error, and downstream format_xxx(None) functions already handle it
("Insufficient data.").

Usage:
    python scripts/update_quant_signals.py

Requires: yfinance, psycopg2, python-dotenv (already in project env)
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, execute_values

from src.tools.market_data import (
    get_stock_snapshot,
    get_risk_inputs,
    get_quality_inputs,
    get_consensus_inputs,
)
from src.quant.valuation_signal import valuation_signal
from src.quant.momentum_signal import momentum_signal
from src.quant.risk_signal import risk_signal
from src.quant.quality_signal import quality_signal
from src.quant.consensus_signal import consensus_signal
from scripts.currency_check import is_usd_reporter

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def _load_target_tickers() -> list[str]:
    universe_path = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"
    with open(universe_path, "r") as f:
        universe = json.load(f)
    return universe["tickers"]


def _to_float(value) -> float | None:
    """
    Converts numpy scalar types (e.g. numpy.float64) to plain Python
    float before they reach psycopg2. json.dumps() (used for the JSONB
    columns via Json()) already handles numpy floats correctly, but
    psycopg2's parameter binding for plain REAL columns does not — it
    passed the raw numpy repr (e.g. "np.float64(0.5)") straight into
    the SQL text, which Postgres then tried to parse as a schema-
    qualified identifier ("np"."float64(...)"), causing every write to
    fail with 'schema "np" does not exist'. round() inside the signal
    engines does not fix this: numpy's round() returns numpy.float64,
    not a Python float.
    """
    return None if value is None else float(value)


def _compute_ticker_row(ticker: str) -> tuple | None:
    """
    Fetches this ticker's inputs and computes all 5 signals, returning
    a single row tuple ready for INSERT. Returns None only if the
    stock_snapshot itself can't be fetched at all (nothing else can be
    computed without it) — individual signals returning None (e.g.
    quality with insufficient history) are still stored as NULL, not
    treated as a ticker-level failure.
    """
    market_data = get_stock_snapshot(ticker)
    if not market_data:
        return None

    risk_inputs      = get_risk_inputs(ticker)
    quality_inputs   = get_quality_inputs(ticker)
    consensus_inputs = get_consensus_inputs(ticker)

    val_result       = valuation_signal(market_data)
    mom_result       = momentum_signal(market_data)
    risk_result      = risk_signal(market_data, risk_inputs)
    quality_result   = quality_signal(quality_inputs)
    consensus_result = consensus_signal(market_data, consensus_inputs)

    valuation_score      = _to_float(val_result.get("valuation_score") if val_result else None)
    momentum_12_1_score  = _to_float(mom_result.get("momentum_12_1_score") if mom_result else None)
    position_52w_score   = _to_float(mom_result.get("position_52w_score") if mom_result else None)

    risk_beta_score     = _to_float((risk_result.get("beta") or {}).get("beta_score") if risk_result else None)
    risk_sharpe_score   = _to_float((risk_result.get("sharpe") or {}).get("sharpe_score") if risk_result else None)
    risk_var_score      = _to_float((risk_result.get("var") or {}).get("var_score") if risk_result else None)
    risk_drawdown_score = _to_float((risk_result.get("max_drawdown") or {}).get("drawdown_score") if risk_result else None)

    quality_score = _to_float(quality_result.get("quality_score") if quality_result else None)

    consensus_recommendation_score = _to_float(consensus_result.get("recommendation_score") if consensus_result else None)
    consensus_upside_score         = _to_float(consensus_result.get("upside_score") if consensus_result else None)
    consensus_trend_score          = _to_float(consensus_result.get("trend_score") if consensus_result else None)

    return (
        ticker,
        Json(val_result) if val_result else None,
        valuation_score,
        Json(mom_result) if mom_result else None,
        momentum_12_1_score,
        position_52w_score,
        Json(risk_result) if risk_result else None,
        risk_beta_score,
        risk_sharpe_score,
        risk_var_score,
        risk_drawdown_score,
        Json(quality_result) if quality_result else None,
        quality_score,
        Json(consensus_result) if consensus_result else None,
        consensus_recommendation_score,
        consensus_upside_score,
        consensus_trend_score,
    )


INSERT_SQL = """
    INSERT INTO quant_signals (
        ticker,
        valuation_data, valuation_score, valuation_computed_at,
        momentum_data, momentum_12_1_score, position_52w_score, momentum_computed_at,
        risk_data, risk_beta_score, risk_sharpe_score, risk_var_score, risk_drawdown_score, risk_computed_at,
        quality_data, quality_score, quality_computed_at,
        consensus_data, consensus_recommendation_score, consensus_upside_score, consensus_trend_score, consensus_computed_at
    )
    VALUES %s
    ON CONFLICT (ticker) DO UPDATE SET
        valuation_data = EXCLUDED.valuation_data,
        valuation_score = EXCLUDED.valuation_score,
        valuation_computed_at = NOW(),
        momentum_data = EXCLUDED.momentum_data,
        momentum_12_1_score = EXCLUDED.momentum_12_1_score,
        position_52w_score = EXCLUDED.position_52w_score,
        momentum_computed_at = NOW(),
        risk_data = EXCLUDED.risk_data,
        risk_beta_score = EXCLUDED.risk_beta_score,
        risk_sharpe_score = EXCLUDED.risk_sharpe_score,
        risk_var_score = EXCLUDED.risk_var_score,
        risk_drawdown_score = EXCLUDED.risk_drawdown_score,
        risk_computed_at = NOW(),
        quality_data = EXCLUDED.quality_data,
        quality_score = EXCLUDED.quality_score,
        quality_computed_at = NOW(),
        consensus_data = EXCLUDED.consensus_data,
        consensus_recommendation_score = EXCLUDED.consensus_recommendation_score,
        consensus_upside_score = EXCLUDED.consensus_upside_score,
        consensus_trend_score = EXCLUDED.consensus_trend_score,
        consensus_computed_at = NOW();
"""
TEMPLATE = (
    "(%s, %s, %s, NOW(), %s, %s, %s, NOW(), %s, %s, %s, %s, %s, NOW(), "
    "%s, %s, NOW(), %s, %s, %s, %s, NOW())"
)


def update_quant_signals():
    print("EquityMind — Quant Signals Updater")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Computing 5 signals per ticker and writing to quant_signals...\n")

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("Removing tickers no longer in the universe...")
    cursor.execute("DELETE FROM quant_signals WHERE ticker != ALL(%s)", (tickers,))
    removed = cursor.rowcount
    conn.commit()
    print(f"  Removed {removed} orphaned ticker(s) not in the current universe.\n")

    success_count = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", flush=True)

        if not is_usd_reporter(ticker):
            print(f"    Skipping {ticker}: non-USD financial reporting")
            failed_tickers.append(ticker)
            continue

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_compute_ticker_row, ticker)
                row = future.result(timeout=30)
        except FutureTimeoutError:
            print(f"    Timed out after 30s for {ticker} — skipping")
            row = None
        except Exception as e:
            print(f"    Computation failed for {ticker}: {e}")
            row = None

        if row is None:
            failed_tickers.append(ticker)
        else:
            try:
                execute_values(cursor, INSERT_SQL, [row], template=TEMPLATE)
                conn.commit()
                success_count += 1
            except Exception as e:
                print(f"    DB write failed for {ticker}: {e}")
                failed_tickers.append(ticker)
                if conn.closed:
                    print("    Connection was closed — reconnecting...")
                    conn = psycopg2.connect(DATABASE_URL)
                    cursor = conn.cursor()
                else:
                    conn.rollback()

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}")

    cursor.close()
    conn.close()

    print(f"\n✓ quant_signals updated — {success_count}/{len(tickers)} tickers succeeded")
    if failed_tickers:
        print(f"  Failed: {failed_tickers}")


if __name__ == "__main__":
    update_quant_signals()
