"""
scripts/update_financial_history.py

Financial History Updater for EquityMind Layer 2 — pulls multi-year
financial statement data (income statement, cash flow, balance sheet)
for every ticker in stock_universe.json and stores it in the
financial_history table (see init_db_financial_history.py), so
questions like "Apple's net income over the last few years" can be
answered from a table lookup instead of the live 2-year-only fetch
that get_quality_inputs() does for signal computation.

Wide format: one row per (ticker, fiscal_year_end), one column per
metric. Chosen over the original long/narrow layout because the
project has a confirmed need for cross-metric filtering within this
table (e.g. "revenue grew but gross margin declined" needs two columns
from the same row — a narrow table can only do this via a self-join).

Data source: yfinance .financials / .cashflow / .balance_sheet.
Confirmed live (2026-07-22, tested across AAPL/MSFT/TSLA/JPM/JNJ/XOM):
yfinance's annual statements return 4 or 5 raw columns depending on
the ticker, but the oldest column is consistently NaN whenever 5 are
returned — so 4 years is the reliable, validated coverage, not 5.

26 metrics across the three statements (see the *_METRICS dicts below)
— all sourced from the same three yfinance calls already made per
ticker for get_quality_inputs(), so this adds no extra API cost beyond
what the project already does. Not every company has every metric
(e.g. banks have no cost_of_revenue) — missing metrics are NULL, not a
placeholder zero.

The INSERT statement's column list is generated from METRIC_COLUMNS at
import time, not hand-written — this guarantees the SQL and the Python
metric dicts can never silently drift out of sync as metrics are added.

Usage:
    python scripts/update_financial_history.py

Requires: yfinance, psycopg2, python-dotenv (already in project env)
"""

import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import yfinance as yf
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from scripts.currency_check import is_usd_reporter

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def _load_target_tickers() -> list[str]:
    """
    Returns the UNION of the current 250-ticker universe and every
    ticker already present in financial_history — not just the current
    universe alone. This table is deliberately append-only: a ticker
    that once entered the universe keeps its accumulated history
    forever, even after it later drops out of the top 250 (e.g. market
    cap decline). Re-including it here on every run means its data
    keeps refreshing/growing for as long as it exists in yfinance,
    rather than being silently abandoned the moment it's no longer in
    the current universe. See update_quant_signals.py for the opposite
    policy (that table IS synced strictly to the current universe,
    since it stores present-day signal scores with no historical value
    once a ticker is no longer relevant).
    """
    universe_path = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"
    with open(universe_path, "r") as f:
        universe = json.load(f)
    current_universe = set(universe["tickers"])

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM financial_history")
    already_tracked = {row[0] for row in cursor.fetchall()}
    cursor.close()
    conn.close()

    return sorted(current_universe | already_tracked)


# yfinance row label -> our column name (snake_case), grouped by statement
INCOME_STATEMENT_METRICS = {
    "Total Revenue":                      "total_revenue",
    "Cost Of Revenue":                    "cost_of_revenue",
    "Gross Profit":                       "gross_profit",
    "Research And Development":           "research_and_development",
    "Selling General And Administration": "selling_general_and_administration",
    "Operating Expense":                  "operating_expense",
    "Operating Income":                   "operating_income",
    "EBIT":                               "ebit",
    "EBITDA":                             "ebitda",
    "Pretax Income":                      "pretax_income",
    "Net Income":                         "net_income",
    "Diluted EPS":                        "diluted_eps",
    "Basic EPS":                          "basic_eps",
}
CASH_FLOW_METRICS = {
    "Operating Cash Flow":         "operating_cash_flow",
    "Capital Expenditure":         "capital_expenditure",
    "Free Cash Flow":              "free_cash_flow",
    "Repurchase Of Capital Stock": "repurchase_of_capital_stock",
    "Cash Dividends Paid":         "cash_dividends_paid",
}
BALANCE_SHEET_METRICS = {
    "Total Assets":                           "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity":                     "stockholders_equity",
    "Cash And Cash Equivalents":               "cash_and_equivalents",
    "Long Term Debt":                          "long_term_debt",
    "Current Assets":                          "current_assets",
    "Current Liabilities":                     "current_liabilities",
    "Ordinary Shares Number":                  "shares_outstanding",
}

ALL_METRICS = {**INCOME_STATEMENT_METRICS, **CASH_FLOW_METRICS, **BALANCE_SHEET_METRICS}
METRIC_COLUMNS = list(ALL_METRICS.values())  # 26 column names, in a fixed order

assert len(METRIC_COLUMNS) == len(set(METRIC_COLUMNS)), \
    "Duplicate column name across the three metric dicts — check for a naming collision."

# Built from METRIC_COLUMNS, not hand-written, so the SQL can never
# silently drift out of sync with the Python metric dicts above.
_ALL_COLUMNS = ["ticker", "fiscal_year_end"] + METRIC_COLUMNS
INSERT_SQL = f"""
    INSERT INTO financial_history ({", ".join(_ALL_COLUMNS)})
    VALUES %s
    ON CONFLICT (ticker, fiscal_year_end) DO UPDATE SET
        {", ".join(f"{col} = EXCLUDED.{col}" for col in METRIC_COLUMNS)},
        updated_at = NOW();
"""


def _extract_ticker_rows(ticker: str) -> list[tuple]:
    """
    Fetches the 3 financial statements for one ticker and merges them
    into one row per fiscal year, with all 26 metrics as columns.
    Metrics not present for a given company/year are left as None
    (SQL NULL), not a placeholder value. Skips the whole ticker
    (returns []) if the statements can't be fetched at all.
    """
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        cf  = stock.cashflow
        bs  = stock.balance_sheet
    except Exception as e:
        print(f"    Skipping {ticker}: {e}")
        return []

    # fiscal_year_end (date) -> {column_name: value}
    by_year = defaultdict(dict)

    def _collect(df, metric_map):
        for row_label, column_name in metric_map.items():
            if row_label not in df.index:
                continue
            for period_end in df.columns:
                value = df.loc[row_label, period_end]
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                by_year[period_end.date()][column_name] = float(value)

    _collect(fin, INCOME_STATEMENT_METRICS)
    _collect(cf, CASH_FLOW_METRICS)
    _collect(bs, BALANCE_SHEET_METRICS)

    rows = []
    for fiscal_year_end, metrics in sorted(by_year.items()):
        row = (ticker, fiscal_year_end) + tuple(metrics.get(col) for col in METRIC_COLUMNS)
        rows.append(row)

    return rows


def update_financial_history():
    print("EquityMind — Financial History Updater")
    print("=" * 50)

    tickers = _load_target_tickers()
    print(f"\nUniverse size: {len(tickers)} tickers")
    print("Fetching financial statements and writing to financial_history...\n")

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    total_rows = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", flush=True)

        if not is_usd_reporter(ticker):
            print(f"    Skipping {ticker}: non-USD financial reporting")
            failed_tickers.append(ticker)
            continue

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_extract_ticker_rows, ticker)
                rows = future.result(timeout=30)
        except FutureTimeoutError:
            print(f"    Timed out after 30s for {ticker} — skipping")
            rows = []
        except Exception as e:
            print(f"    Extraction failed for {ticker}: {e}")
            rows = []

        if not rows:
            failed_tickers.append(ticker)
        else:
            try:
                execute_values(cursor, INSERT_SQL, rows)
                conn.commit()
                total_rows += len(rows)
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

    print(f"\n✓ financial_history updated — {total_rows} rows written across "
          f"{len(tickers) - len(failed_tickers)}/{len(tickers)} tickers")
    if failed_tickers:
        print(f"  Skipped (no data): {failed_tickers}")


if __name__ == "__main__":
    update_financial_history()
