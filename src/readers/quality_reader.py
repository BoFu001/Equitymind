"""
src/readers/quality_reader.py

Reads Quality signal inputs from the financial_history table (see
scripts/init_db_financial_history.py / update_financial_history.py)
instead of calling yfinance live — replaces get_quality_inputs()
(market_data.py), which was making a redundant live call for the same
26 metrics update_financial_history.py already fetches and stores.

Renamed from financial_history_reader.py on 2026-07-27 — that name
became ambiguous once a second, unrelated reader
(financial_history_reader.py, for general historical-trend display,
not tied to any single signal) was introduced. This file's name now
matches the signal it serves (Quality), consistent with the project's
naming convention for the other signal-specific readers.

Design decision (2026-07-27): reads ONLY from financial_history, with
no live-yfinance fallback if a ticker isn't in the table. This applies
uniformly whether quality_signal() is currently using annual
comparison or (later) T12M — a live fallback would only ever be
possible for the annual case (yfinance can supply ~4-5 years of
annual data on demand), not for T12M (which needs 8 quarters of
history that yfinance itself doesn't expose beyond ~5-7 periods —
see project notes, 2026-07-26). Making annual "fall back to live" and
T12M "never fall back" would mean the same ticker's behavior silently
changes once the project switches to T12M months from now — a
confusing, time-dependent inconsistency. Returning "insufficient
data" uniformly for any ticker not in financial_history, regardless
of which comparison method is active, is simpler and predictable.

Currently reads the two most recent 'annual' period_type rows. Once
the project switches to T12M (after enough tickers accumulate 8+
quarterly periods — see update_financial_history.py), this function's
query changes to sum trailing quarters instead — quality_signal()
itself does not need to change, since it only cares about receiving
{"current": {...}, "prior": {...}} in the same shape either way.

TODO (2026-07-27): check back around 2027-04-27 (~9 months from
today) — most tickers should have accumulated 8+ quarterly periods in
financial_history by then (quarterly data grows ~1 period every 3
months; a handful of tickers were at 5 periods on 2026-07-26). Run:
    SELECT ticker, COUNT(*) FROM financial_history
    WHERE period_type = 'quarterly' GROUP BY ticker HAVING COUNT(*) < 8;
If most/all tickers clear 8 periods, switch this function to T12M
(sum of trailing 4 quarters vs. the 4 quarters before that) — this
was deliberately deferred as a single, universe-wide cutover (not a
per-ticker gradual switch) to keep quality_score comparable across
all tickers at any given point in time (see project notes,
2026-07-26 — annual and T12M measure different windows and are not
directly comparable, so mixing them within the universe at once
would make cross-ticker Quality comparisons meaningless).
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# The 9 fields quality_signal()'s F-Score calculations actually read
# from current/prior — confirmed 2026-07-27 by grepping quality_signal.py
# directly, not assumed. All 9 already exist as financial_history columns.
_QUALITY_FIELDS = [
    "net_income",
    "operating_cash_flow",
    "total_assets",
    "long_term_debt",
    "current_assets",
    "current_liabilities",
    "shares_outstanding",
    "gross_profit",
    "total_revenue",
]


def get_quality_inputs_from_db(ticker: str) -> dict | None:
    """
    Returns {"current": {...}, "prior": {...}} in the same shape
    get_quality_inputs() (market_data.py, now unused by quality_signal)
    used to produce — quality_signal() itself needs no changes.

    Returns None if fewer than 2 annual periods exist for this ticker
    in financial_history (mirrors get_quality_inputs()'s own "need at
    least 2 fiscal years" requirement) — this is the ONLY degradation
    path; there is no live-yfinance fallback (see module docstring).
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    columns = ", ".join(_QUALITY_FIELDS)
    cursor.execute(
        f"""
        SELECT period_end, {columns}
        FROM financial_history
        WHERE ticker = %s AND period_type = 'annual'
        ORDER BY period_end DESC
        LIMIT 2
        """,
        (ticker,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(rows) < 2:
        return None

    def _row_to_dict(row) -> dict:
        # row[0] is period_end, the rest align with _QUALITY_FIELDS in order.
        # PostgreSQL NUMERIC columns come back as Python Decimal via
        # psycopg2, not float — quality_signal.py's arithmetic (e.g.
        # ni/1e9) assumes float, matching what yfinance always returned.
        # Converting here means quality_signal() never needs to know or
        # care which source (live yfinance vs this DB read) produced
        # the numbers it's given.
        return {
            field: (float(value) if value is not None else None)
            for field, value in zip(_QUALITY_FIELDS, row[1:])
        }

    current_row, prior_row = rows[0], rows[1]

    return {
        "current": _row_to_dict(current_row),
        "prior":   _row_to_dict(prior_row),
        # The period_end this "current" data is drawn from — lets
        # update_quant_signals.py record which financial_history period
        # a cached Quality result was computed from (quant_signals.
        # quality_period_end), so it can skip recomputing when nothing
        # newer has been filed since.
        "current_period_end": current_row[0],
    }
