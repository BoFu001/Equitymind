"""
scripts/update_financial_history.py

Financial History Updater for EquityMind Layer 2 — pulls both annual
and quarterly financial statement data (income statement, cash flow,
balance sheet) for every ticker in the stock_universe table and
stores it in financial_history (see init_db_financial_history.py).

Annual and quarterly share one loop, one metric mapping, and one
INSERT template — only the three source calls and the period_type tag
differ. Both fetch unconditionally on every run, with no check for
whether the data has changed.

That check is absent by design. yfinance creates the three statements'
columns for a new period at different times: the income statement gets
a column holding only EPS as soon as results are announced, while the
balance sheet and cash flow statement have no column for that period
at all. Comparing the newest period_end against what is stored would
therefore record a period as present while nearly every figure in it
is still missing, and never revisit it. Nor can a stored period be
tested for completeness — field counts vary legitimately between
periods, since a quarter with no impairment or restructuring has no
row for it and a fiscal year-end discloses fixed-asset and pension
detail a quarter does not. Overwriting everything each run avoids both
problems. The table is a mirror of the source: the tickers are those
in stock_universe, the periods are those yfinance currently returns,
and rows outside either set are deleted rather than kept.

yfinance is an unofficial wrapper with no data dictionary and no
update guarantees, so this is a stopgap: the logic here would be
rewritten against a source exposing an explicit last-updated
timestamp.

Quarterly data feeds a future T12M (trailing-twelve-month) Quality
signal. quarterly_financials/_cashflow/_balance_sheet return 5-7
periods depending on the ticker, and quarterly figures are discrete,
not year-to-date cumulative.

period_type ('annual'/'quarterly') is written explicitly for every
row — the column has no default (see init_db_financial_history.py) —
so a missing value fails loudly rather than being silently guessed.

33 metrics across the three statements, six yfinance calls per
ticker per run (three statements each for annual and quarterly).

The INSERT statement's column list is generated from METRIC_COLUMNS at
import time, not hand-written — this guarantees the SQL and the Python
metric dicts can never silently drift out of sync as metrics are added.

Usage:
    python scripts/update_financial_history.py
"""

import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import yfinance as yf
from psycopg2.extras import execute_values
from config import DATABASE_URL, PIPELINE_THROTTLE_SECONDS


def _load_target_tickers() -> list[str]:
    """
    Returns the current universe. Rows for tickers outside it are
    deleted at the start of each run.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM stock_universe ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tickers


# yfinance row label -> our column name (snake_case), grouped by
# statement. Annual and quarterly use identical row labels, so one
# mapping serves both.
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
METRIC_COLUMNS = list(ALL_METRICS.values())  # 33 column names, in a fixed order

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
    Assembles one row per period from the three statement DataFrames.
    Metrics not present for a given company/period are left as None
    (SQL NULL), not a placeholder value.
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
    Fetches the three ANNUAL statements for one ticker. Returns []
    if they can't be fetched at all; the quarterly pass still runs.
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
    Fetches the three QUARTERLY statements for one ticker. Returns []
    if they can't be fetched at all; the annual pass still runs.
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


def update_financial_history():
    print("EquityMind — Financial History Updater")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Fetching annual and quarterly statements...\n")

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Rows for tickers that have left the universe are dropped rather
    # than left behind — this table holds what the current universe
    # looks like now, not an accumulated record.
    cursor.execute("DELETE FROM financial_history WHERE ticker != ALL(%s)", (tickers,))
    if cursor.rowcount:
        print(f"Removed {cursor.rowcount} row(s) for tickers no longer in the universe.")
    conn.commit()

    total_rows = 0
    annual_failed = []
    quarterly_failed = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", flush=True)

        for period_type, extractor, failed in (
            ("annual", _extract_annual_rows, annual_failed),
            ("quarterly", _extract_quarterly_rows, quarterly_failed),
        ):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(extractor, ticker)
                    rows = future.result(timeout=30)
            except FutureTimeoutError:
                print(f"    Timed out after 30s for {ticker} ({period_type}) — skipping")
                rows = []
            except Exception as e:
                print(f"    Extraction failed for {ticker} ({period_type}): {e}")
                rows = []

            if not rows:
                failed.append(ticker)
            else:
                try:
                    execute_values(cursor, INSERT_SQL, rows)
                    # Periods yfinance no longer offers are dropped, so
                    # the table mirrors the source rather than keeping
                    # what it used to hold.
                    cursor.execute(
                        """
                        DELETE FROM financial_history
                        WHERE ticker = %s AND period_type = %s
                          AND period_end != ALL(%s)
                        """,
                        (ticker, period_type, [r[1] for r in rows]),
                    )
                    conn.commit()
                    total_rows += len(rows)
                except Exception as e:
                    print(f"    DB write failed for {ticker} ({period_type}): {e}")
                    failed.append(ticker)
                    if conn.closed:
                        print("    Connection was closed — reconnecting...")
                        conn = psycopg2.connect(DATABASE_URL)
                        cursor = conn.cursor()
                    else:
                        conn.rollback()

        # Six yfinance calls per ticker, and this API rate-limits.
        if PIPELINE_THROTTLE_SECONDS > 0:
            time.sleep(PIPELINE_THROTTLE_SECONDS)

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(tickers)}")

    cursor.close()
    conn.close()

    print(f"\n✓ financial_history updated — {total_rows} rows written across {len(tickers)} tickers")
    if annual_failed:
        print(f"  Annual — failed: {annual_failed}")
    if quarterly_failed:
        print(f"  Quarterly — failed: {quarterly_failed}")


if __name__ == "__main__":
    update_financial_history()
