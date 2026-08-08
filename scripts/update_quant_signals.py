"""
scripts/update_quant_signals.py

Quant Signal Updater for EquityMind Layer 2 — computes all 6 cacheable
signals (Valuation, Momentum, Risk, Quality, Consensus, Short — News
Sentiment is deliberately excluded, see quant_signals table docstring)
for every ticker in the stock_universe table and stores them in the
quant_signals table (see init_db_quant_signals.py).

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
momentum_signal() for a ticker missing from momentum_benchmarks table)
are stored as NULL — this is a legitimate, expected outcome, not an
error, and downstream format_xxx(None) functions already handle it
("Insufficient data.").

Usage:
    python scripts/update_quant_signals.py

Requires: yfinance, psycopg2, python-dotenv (already in project env)
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2

from config import PIPELINE_THROTTLE_SECONDS
from psycopg2.extras import Json, execute_values

from src.readers.snapshot_reader import get_stock_snapshot
from src.readers.valuation_reader import get_valuation_inputs
from src.readers.momentum_reader import get_momentum_inputs
from src.readers.risk_reader import get_risk_inputs
from src.readers.consensus_reader import get_consensus_snapshot, get_consensus_trend
from src.readers.quality_reader import get_quality_inputs_from_db
from src.readers.short_reader import get_short_inputs
from src.quant.valuation_signal import valuation_signal
from src.quant.momentum_signal import momentum_signal
from src.quant.risk_signal import risk_signal
from src.quant.quality_signal import quality_signal
from src.quant.consensus_signal import consensus_signal
from src.quant.short_signal import short_signal
from config import DATABASE_URL



def _load_target_tickers() -> list[str]:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM stock_universe")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tickers


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


def _quality_needs_recompute(cursor, ticker: str) -> bool:
    """
    Compares financial_history's latest annual period_end for this
    ticker against quant_signals.quality_period_end (the period the
    cached Quality result was last computed from). Returns True if
    Quality should be recomputed — either because there's no cached
    result yet, or because a newer annual report has been filed since.

    Returns True (recompute) on any ambiguity — e.g. financial_history
    has no annual data at all for this ticker (quality_signal() will
    correctly return None downstream; there's nothing to "skip" here),
    or the ticker has no row in quant_signals yet (first run).
    """
    cursor.execute(
        "SELECT MAX(period_end) FROM financial_history "
        "WHERE ticker = %s AND period_type = 'annual'",
        (ticker,),
    )
    row = cursor.fetchone()
    latest_annual_period_end = row[0] if row else None
    if latest_annual_period_end is None:
        return True

    cursor.execute(
        "SELECT quality_period_end FROM quant_signals WHERE ticker = %s",
        (ticker,),
    )
    row = cursor.fetchone()
    cached_period_end = row[0] if row else None
    if cached_period_end is None:
        return True

    return latest_annual_period_end > cached_period_end


def _compute_ticker_row(ticker: str, skip_quality: bool = False) -> tuple | None:
    """
    Fetches this ticker's inputs and computes signals, returning a row
    tuple ready for INSERT. Returns None only if the stock_snapshot
    itself can't be fetched at all (nothing else can be computed
    without it) — individual signals returning None (e.g. quality with
    insufficient history) are still stored as NULL, not treated as a
    ticker-level failure.

    skip_quality: when True, Quality is NOT recomputed and is omitted
    entirely from the returned tuple — see _quality_needs_recompute()
    and INSERT_SQL_SKIP_QUALITY. This is decided by the caller (which
    already has the DB cursor needed to check financial_history vs.
    quant_signals.quality_period_end), not by this function — this
    function only knows how to skip, not why.
    """
    snapshot = get_stock_snapshot(ticker)
    if not snapshot:
        return None

    # valuation_signal() fetched independently of snapshot as of
    # 2026-07-27 -- see valuation_reader.py for why (same principle
    # already applied to consensus_reader.py).
    valuation_inputs = get_valuation_inputs(ticker)
    momentum_inputs = get_momentum_inputs(ticker)

    risk_inputs = get_risk_inputs(ticker)

    # consensus_signal() needs both pieces merged into one dict -- see
    # consensus_reader.py and consensus_signal.py for why these are
    # fetched independently of snapshot rather than shared with it.
    consensus_snapshot = get_consensus_snapshot(ticker)
    consensus_trend = get_consensus_trend(ticker)
    consensus_data = (
        {"snapshot": consensus_snapshot, "trend": consensus_trend}
        if consensus_snapshot else None
    )

    short_inputs = get_short_inputs(ticker)

    val_result        = valuation_signal(valuation_inputs) if valuation_inputs else None
    mom_result        = momentum_signal(momentum_inputs)
    risk_result       = risk_signal(risk_inputs)
    consensus_result  = consensus_signal(consensus_data)
    short_result      = short_signal(short_inputs)

    valuation_score      = _to_float(val_result.get("valuation_score") if val_result else None)
    momentum_12_1_score  = _to_float(mom_result.get("momentum_12_1_score") if mom_result else None)
    position_52w_score   = _to_float(mom_result.get("position_52w_score") if mom_result else None)

    risk_beta_score     = _to_float((risk_result.get("beta") or {}).get("beta_score") if risk_result else None)
    risk_sharpe_score   = _to_float((risk_result.get("sharpe") or {}).get("sharpe_score") if risk_result else None)
    risk_var_score      = _to_float((risk_result.get("var") or {}).get("var_score") if risk_result else None)
    risk_drawdown_score = _to_float((risk_result.get("max_drawdown") or {}).get("drawdown_score") if risk_result else None)

    consensus_recommendation_score = _to_float(consensus_result.get("recommendation_score") if consensus_result else None)
    consensus_upside_score         = _to_float(consensus_result.get("upside_score") if consensus_result else None)
    consensus_trend_score          = _to_float(consensus_result.get("trend_score") if consensus_result else None)

    short_interest_pct  = _to_float(short_result.get("short_interest_pct") if short_result else None)
    days_to_cover       = _to_float(short_result.get("days_to_cover") if short_result else None)

    row = (
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
    )

    if not skip_quality:
        # Reads from financial_history (DB) instead of calling yfinance
        # live — see src/readers/quality_reader.py for why.
        quality_inputs = get_quality_inputs_from_db(ticker)
        quality_result = quality_signal(quality_inputs)
        quality_score = _to_float(quality_result.get("quality_score") if quality_result else None)
        quality_period_end = quality_inputs["current_period_end"] if quality_inputs else None
        row = row + (
            Json(quality_result) if quality_result else None,
            quality_score,
            quality_period_end,
        )

    row = row + (
        Json(consensus_result) if consensus_result else None,
        consensus_recommendation_score,
        consensus_upside_score,
        consensus_trend_score,
        Json(short_result) if short_result else None,
        short_interest_pct,
        days_to_cover,
    )

    return row


# Used when Quality WAS recomputed this run (skip_quality=False) —
# writes all 6 signals, including the 3 quality_* columns.
INSERT_SQL_WITH_QUALITY = """
    INSERT INTO quant_signals (
        ticker,
        valuation_data, valuation_score, valuation_computed_at,
        momentum_data, momentum_12_1_score, position_52w_score, momentum_computed_at,
        risk_data, risk_beta_score, risk_sharpe_score, risk_var_score, risk_drawdown_score, risk_computed_at,
        quality_data, quality_score, quality_period_end, quality_computed_at,
        consensus_data, consensus_recommendation_score, consensus_upside_score, consensus_trend_score, consensus_computed_at,
        short_data, short_interest_pct, days_to_cover, short_computed_at
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
        quality_period_end = EXCLUDED.quality_period_end,
        quality_computed_at = NOW(),
        consensus_data = EXCLUDED.consensus_data,
        consensus_recommendation_score = EXCLUDED.consensus_recommendation_score,
        consensus_upside_score = EXCLUDED.consensus_upside_score,
        consensus_trend_score = EXCLUDED.consensus_trend_score,
        consensus_computed_at = NOW(),
        short_data = EXCLUDED.short_data,
        short_interest_pct = EXCLUDED.short_interest_pct,
        days_to_cover = EXCLUDED.days_to_cover,
        short_computed_at = NOW();
"""
TEMPLATE_WITH_QUALITY = (
    "(%s, %s, %s, NOW(), %s, %s, %s, NOW(), %s, %s, %s, %s, %s, NOW(), "
    "%s, %s, %s, NOW(), %s, %s, %s, %s, NOW(), %s, %s, %s, NOW())"
)

# Used when Quality was SKIPPED this run (skip_quality=True) — the
# quality_* columns are entirely absent from both the column list and
# the SET clause, so ON CONFLICT DO UPDATE leaves them untouched
# rather than overwriting them with NULL.
INSERT_SQL_SKIP_QUALITY = """
    INSERT INTO quant_signals (
        ticker,
        valuation_data, valuation_score, valuation_computed_at,
        momentum_data, momentum_12_1_score, position_52w_score, momentum_computed_at,
        risk_data, risk_beta_score, risk_sharpe_score, risk_var_score, risk_drawdown_score, risk_computed_at,
        consensus_data, consensus_recommendation_score, consensus_upside_score, consensus_trend_score, consensus_computed_at,
        short_data, short_interest_pct, days_to_cover, short_computed_at
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
        consensus_data = EXCLUDED.consensus_data,
        consensus_recommendation_score = EXCLUDED.consensus_recommendation_score,
        consensus_upside_score = EXCLUDED.consensus_upside_score,
        consensus_trend_score = EXCLUDED.consensus_trend_score,
        consensus_computed_at = NOW(),
        short_data = EXCLUDED.short_data,
        short_interest_pct = EXCLUDED.short_interest_pct,
        days_to_cover = EXCLUDED.days_to_cover,
        short_computed_at = NOW();
"""
TEMPLATE_SKIP_QUALITY = (
    "(%s, %s, %s, NOW(), %s, %s, %s, NOW(), %s, %s, %s, %s, %s, NOW(), "
    "%s, %s, %s, %s, NOW(), %s, %s, %s, NOW())"
)


def update_quant_signals():
    """
    See the throttle comment inside the main loop below for why
    PIPELINE_THROTTLE_SECONDS exists.
    """
    print("EquityMind — Quant Signals Updater")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Computing 6 signals per ticker and writing to quant_signals...\n")

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("Removing tickers no longer in the universe...")
    cursor.execute("DELETE FROM quant_signals WHERE ticker != ALL(%s)", (tickers,))
    removed = cursor.rowcount
    conn.commit()
    print(f"  Removed {removed} orphaned ticker(s) not in the current universe.\n")

    success_count = 0
    quality_skipped_count = 0
    quality_recomputed_tickers = []
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", flush=True)

        # Throttle for Railway only — avoids yfinance rate limits.
        if PIPELINE_THROTTLE_SECONDS > 0:
            time.sleep(PIPELINE_THROTTLE_SECONDS)

        skip_quality = not _quality_needs_recompute(cursor, ticker)
        if skip_quality:
            quality_skipped_count += 1
        else:
            quality_recomputed_tickers.append(ticker)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_compute_ticker_row, ticker, skip_quality)
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
                if skip_quality:
                    execute_values(cursor, INSERT_SQL_SKIP_QUALITY, [row], template=TEMPLATE_SKIP_QUALITY)
                else:
                    execute_values(cursor, INSERT_SQL_WITH_QUALITY, [row], template=TEMPLATE_WITH_QUALITY)
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
    print(f"  Quality — skipped (no new annual report): {quality_skipped_count}")
    print(f"  Quality — recomputed: {len(quality_recomputed_tickers)}")
    if quality_recomputed_tickers:
        print(f"    {quality_recomputed_tickers}")
    if failed_tickers:
        print(f"  Failed: {failed_tickers}")


if __name__ == "__main__":
    update_quant_signals()
