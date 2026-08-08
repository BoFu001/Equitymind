"""
scripts/update_financial_history.py

Financial History Updater for EquityMind Layer 2 — pulls both annual
and quarterly financial statement data (income statement, cash flow,
balance sheet) for every ticker in the stock_universe table and
stores it in financial_history (see init_db_financial_history.py).

Two independent write paths, sharing the same 26-metric mapping,
database connection, retry/timeout handling, and INSERT template.
BOTH check for new data before writing — companies file annually
(10-K) and quarterly (10-Q), so re-fetching an unchanged statement on
every run is pure waste on both sides, not just the quarterly one.

The freshness check (_has_new_data) compares this ticker's latest
period_end already in financial_history (for the given
period_type) against what yfinance currently offers; skips the full
statement fetch entirely if nothing new has been filed since last
run. yfinance rate-limiting (observed repeatedly, 2026-07-2X) makes
avoiding unnecessary calls a real, non-trivial benefit.

Quarterly accumulation is the mechanism toward a T12M
(trailing-twelve-month) Quality signal (see project notes,
2026-07-26): quarterly history grows by one period roughly every 3
months as new 10-Qs are filed. Confirmed via live yfinance test
(2026-07-26, 33 companies across ~15 sectors): quarterly_financials/
_cashflow/_balance_sheet return 5-7 periods depending on the ticker
(never fewer than 5); quarterly figures are discrete (not
year-to-date cumulative) — cross-checked against Apple's own
published FY2025 Q1 operating cash flow ($29.935B, 0.03% deviation
from inferred value) and JPM's FY2025 revenue (four quarters summed
vs annual total, 0.32% deviation).

period_type ('annual'/'quarterly') is written explicitly for every
row — the column has no default (see init_db_financial_history.py) —
so a missing value fails loudly rather than being silently guessed.

33 metrics across the three statements — all sourced from the same
three yfinance calls already made per ticker for get_quality_inputs(),
so this adds no extra API cost beyond what the project already does.

The INSERT statement's column list is generated from METRIC_COLUMNS at
import time, not hand-written — this guarantees the SQL and the Python
metric dicts can never silently drift out of sync as metrics are added.

Usage:
    python scripts/update_financial_history.py

Requires: yfinance, psycopg2, python-dotenv (already in project env)
"""

import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import yfinance as yf
from psycopg2.extras import execute_values
from config import DATABASE_URL




def _load_target_tickers() -> list[str]:
    """
    Returns the UNION of the current 250-ticker universe and every
    ticker already present in financial_history — not just the current
    universe alone. financial_history is append-only: a ticker that
    once entered the universe keeps its accumulated history forever,
    even after it later drops out of the top 250. See
    update_quant_signals.py for the opposite policy (that table IS
    synced strictly to the current universe).
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("SELECT ticker FROM stock_universe")
    current_universe = {row[0] for row in cursor.fetchall()}

    cursor.execute("SELECT DISTINCT ticker FROM financial_history")
    already_tracked = {row[0] for row in cursor.fetchall()}

    cursor.close()
    conn.close()

    return sorted(current_universe | already_tracked)


# yfinance row label -> our column name (snake_case), grouped by statement.
# Confirmed identical row labels for annual and quarterly statements
# (2026-07-24: quarterly has zero fields not also in annual — quarterly
# is a strict subset), so the same mapping serves both writers.
INCOME_STATEMENT_METRICS = {
    "Total Revenue":                          "total_revenue",
    "Cost Of Revenue":                        "cost_of_revenue",
    "Gross Profit":                           "gross_profit",
    "Research And Development":               "research_and_development",
    "Selling General And Administration":     "selling_general_and_administration",
    "Operating Expense":                      "operating_expense",
    "Operating Income":                       "operating_income",
    "EBIT":                                   "ebit",
    "EBITDA":                                 "ebitda",
    "Pretax Income":                          "pretax_income",
    "Net Income":                             "net_income",
    "Diluted EPS":                            "diluted_eps",
    "Basic EPS":                              "basic_eps",
    "Interest Expense":                       "interest_expense",
}
CASH_FLOW_METRICS = {
    "Operating Cash Flow":                     "operating_cash_flow",
    "Capital Expenditure":                     "capital_expenditure",
    "Free Cash Flow":                          "free_cash_flow",
    "Repurchase Of Capital Stock":             "repurchase_of_capital_stock",
    "Cash Dividends Paid":                     "cash_dividends_paid",
    "Depreciation Amortization Depletion":     "depreciation_amortization_depletion",
}
BALANCE_SHEET_METRICS = {
    "Total Assets":                            "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity":                     "stockholders_equity",
    "Cash And Cash Equivalents":               "cash_and_equivalents",
    "Long Term Debt":                          "long_term_debt",
    "Current Assets":                          "current_assets",
    "Current Liabilities":                     "current_liabilities",
    "Ordinary Shares Number":                  "shares_outstanding",
    "Retained Earnings":                       "retained_earnings",
    "Net PPE":                                 "net_ppe",
    "Accounts Receivable":                     "accounts_receivable",
    "Inventory":                               "inventory",
    "Total Debt":                              "total_debt",
}

ALL_METRICS = {**INCOME_STATEMENT_METRICS, **CASH_FLOW_METRICS, **BALANCE_SHEET_METRICS}
METRIC_COLUMNS = list(ALL_METRICS.values())  # 26 column names, in a fixed order

assert len(METRIC_COLUMNS) == len(set(METRIC_COLUMNS)), \
    "Duplicate column name across the three metric dicts — check for a naming collision."

_ALL_COLUMNS = ["ticker", "period_end", "period_type"] + METRIC_COLUMNS
INSERT_SQL = f"""
    INSERT INTO financial_history ({", ".join(_ALL_COLUMNS)})
    VALUES %s
    ON CONFLICT (ticker, period_end, period_type) DO UPDATE SET
        {", ".join(f"{col} = EXCLUDED.{col}" for col in METRIC_COLUMNS)},
        updated_at = NOW();
"""


def _extract_rows(fin, cf, bs, ticker: str, period_type: str) -> list[tuple]:
    """
    Shared row-assembly logic for both annual and quarterly statements
    (same three DataFrames, same 26-metric mapping — only the source
    call and the period_type tag differ between callers). Metrics not
    present for a given company/period are left as None (SQL NULL),
    not a placeholder value.
    """
    by_period = defaultdict(dict)

    def _collect(df, metric_map):
        for row_label, column_name in metric_map.items():
            if row_label not in df.index:
                continue
            for period_end in df.columns:
                value = df.loc[row_label, period_end]
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                by_period[period_end.date()][column_name] = float(value)

    _collect(fin, INCOME_STATEMENT_METRICS)
    _collect(cf, CASH_FLOW_METRICS)
    _collect(bs, BALANCE_SHEET_METRICS)

    rows = []
    for period_end, metrics in sorted(by_period.items()):
        row = (ticker, period_end, period_type) + tuple(metrics.get(col) for col in METRIC_COLUMNS)
        rows.append(row)

    return rows


def _extract_annual_rows(ticker: str) -> list[tuple]:
    """
    Fetches the 3 ANNUAL financial statements for one ticker. Skips
    the whole ticker (returns []) if the statements can't be fetched
    at all.
    """
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        cf  = stock.cashflow
        bs  = stock.balance_sheet
    except Exception as e:
        print(f"    Skipping {ticker} (annual): {e}")
        return []

    return _extract_rows(fin, cf, bs, ticker, "annual")


def _extract_quarterly_rows(ticker: str) -> list[tuple]:
    """
    Fetches the 3 QUARTERLY financial statements for one ticker. Skips
    the whole ticker (returns []) if the statements can't be fetched
    at all.
    """
    try:
        stock = yf.Ticker(ticker)
        fin = stock.quarterly_financials
        cf  = stock.quarterly_cashflow
        bs  = stock.quarterly_balance_sheet
    except Exception as e:
        print(f"    Skipping {ticker} (quarterly): {e}")
        return []

    return _extract_rows(fin, cf, bs, ticker, "quarterly")


def _latest_period_in_db(cursor, ticker: str, period_type: str):
    """
    Returns the most recent period_end already stored for this
    ticker under the given period_type ('annual' or 'quarterly'), or
    None if none exist yet.
    """
    cursor.execute(
        """
        SELECT MAX(period_end) FROM financial_history
        WHERE ticker = %s AND period_type = %s
        """,
        (ticker, period_type),
    )
    return cursor.fetchone()[0]


def _has_new_data(ticker: str, latest_in_db, quarterly: bool) -> bool:
    """
    Cheap check: does yfinance currently offer a period (annual or
    quarterly, per the `quarterly` flag) newer than what we already
    have? Only reads the column index (dates), not the full statement
    — this "check before you fetch" step applies equally to annual
    (companies file once a year — daily re-fetching is pure waste)
    and quarterly data.

    Returns True (fetch) if there's nothing in the DB yet, or if
    yfinance's latest available period is newer than what's stored.
    """
    if latest_in_db is None:
        return True
    try:
        stock = yf.Ticker(ticker)
        df = stock.quarterly_financials if quarterly else stock.financials
        latest_available = df.columns[0].date()
        return latest_available > latest_in_db
    except Exception:
        # If the cheap check itself fails, fall back to fetching —
        # safer to do the expensive call than to silently skip forever.
        return True


def update_financial_history():
    print("EquityMind — Financial History Updater")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Checking for new annual and quarterly filings...\n")

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    total_rows = 0
    annual_failed = []
    quarterly_failed = []
    annual_fetched = []
    quarterly_fetched = []
    annual_skipped = 0
    quarterly_skipped = 0

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", flush=True)

        # ── Annual: check for new data first, skip if nothing new ──
        annual_latest_in_db = _latest_period_in_db(cursor, ticker, "annual")
        if not _has_new_data(ticker, annual_latest_in_db, quarterly=False):
            annual_skipped += 1
        else:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_extract_annual_rows, ticker)
                    annual_rows = future.result(timeout=30)
            except FutureTimeoutError:
                print(f"    Timed out after 30s for {ticker} (annual) — skipping")
                annual_rows = []
            except Exception as e:
                print(f"    Extraction failed for {ticker} (annual): {e}")
                annual_rows = []

            if not annual_rows:
                annual_failed.append(ticker)
            else:
                try:
                    execute_values(cursor, INSERT_SQL, annual_rows)
                    conn.commit()
                    total_rows += len(annual_rows)
                    annual_fetched.append(ticker)
                except Exception as e:
                    print(f"    DB write failed for {ticker} (annual): {e}")
                    annual_failed.append(ticker)
                    if conn.closed:
                        print("    Connection was closed — reconnecting...")
                        conn = psycopg2.connect(DATABASE_URL)
                        cursor = conn.cursor()
                    else:
                        conn.rollback()

        # ── Quarterly: check for new data first, skip if nothing new ──
        quarterly_latest_in_db = _latest_period_in_db(cursor, ticker, "quarterly")
        if not _has_new_data(ticker, quarterly_latest_in_db, quarterly=True):
            quarterly_skipped += 1
        else:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_extract_quarterly_rows, ticker)
                    quarterly_rows = future.result(timeout=30)
            except FutureTimeoutError:
                print(f"    Timed out after 30s for {ticker} (quarterly) — skipping")
                quarterly_rows = []
            except Exception as e:
                print(f"    Extraction failed for {ticker} (quarterly): {e}")
                quarterly_rows = []

            if not quarterly_rows:
                quarterly_failed.append(ticker)
            else:
                try:
                    execute_values(cursor, INSERT_SQL, quarterly_rows)
                    conn.commit()
                    total_rows += len(quarterly_rows)
                    quarterly_fetched.append(ticker)
                except Exception as e:
                    print(f"    DB write failed for {ticker} (quarterly): {e}")
                    quarterly_failed.append(ticker)
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

    print(f"\n✓ financial_history updated — {total_rows} rows written across {len(tickers)} tickers")
    print(f"  Annual — skipped (no new data): {annual_skipped}")
    print(f"  Annual — fetched: {annual_fetched if annual_fetched else 'none'}")
    if annual_failed:
        print(f"  Annual — failed: {annual_failed}")
    print(f"  Quarterly — skipped (no new data): {quarterly_skipped}")
    print(f"  Quarterly — fetched: {quarterly_fetched if quarterly_fetched else 'none'}")
    if quarterly_failed:
        print(f"  Quarterly — failed: {quarterly_failed}")


if __name__ == "__main__":
    update_financial_history()
