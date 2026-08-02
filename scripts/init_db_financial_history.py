"""
scripts/init_db_financial_history.py

One-time database setup script for EquityMind's multi-year financial
history store — creates the financial_history table in the same
PostgreSQL database used for sec_chunks and quant_signals.

Wide format: one row per (ticker, period_end, period_type), one
column per metric — chosen over a long/narrow (ticker, period_end,
metric_name, value) layout because the project has a confirmed need
for cross-metric filtering within this table (e.g. "companies where
revenue grew but gross margin declined" needs two columns from the
same row, which a narrow table can only do via a self-join). This
mirrors the design of quant_signals, which uses the same reasoning.

period_type ('annual' | 'quarterly') was added to the primary key on
2026-07-26 to support quarterly data accumulation toward a T12M
(trailing-twelve-month) Quality signal — a ticker's Q4 period_end
is often the exact same calendar date as its annual period_end
(e.g. AAPL: both 2025-09-30), so without period_type in the key, an
annual and a Q4 quarterly row for the same date would collide.
period_type has NO DEFAULT and is NOT NULL — both the annual writer
(update_financial_history.py) and the future quarterly writer always
know which kind of data they are writing, so there is no legitimate
case for omitting it; a missing value should fail loudly, not be
silently guessed.

Confirmed via live yfinance test (2026-07-22, tested across
AAPL/MSFT/TSLA/JPM/JNJ/XOM): yfinance's annual statements return 4 or
5 raw columns depending on the ticker, but the oldest column is
consistently NaN whenever 5 are returned — so 4 years is the reliable,
validated coverage, not 5. Coverage grows by one fiscal year annually
as new annual reports are filed; there is no fixed cap, and the system
should state actual coverage to users rather than implying a fixed
history length.

33 metrics across the three statements (income statement, cash flow,
balance sheet) — all sourced from the same three yfinance calls
already made per ticker for get_quality_inputs(), so this adds no
extra API cost beyond what the project already does. Not every
company has every metric (e.g. banks have no cost_of_revenue; many
growth stocks pay no dividends) — missing metrics are NULL, not a
placeholder zero.

Uses CREATE TABLE IF NOT EXISTS — safe to run repeatedly without
destroying existing data. (This table was migrated once from an
earlier long/narrow schema to this wide schema on 2026-07-22, which
did require a one-time DROP; that migration is complete.)

Run once to create the table:
    python scripts/init_db_financial_history.py
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def init():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("Creating financial_history table (wide format)...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_history (
            ticker                              TEXT NOT NULL,
            period_end                          DATE NOT NULL,

            -- Income statement (14 columns)
            total_revenue                       NUMERIC,
            cost_of_revenue                     NUMERIC,
            gross_profit                        NUMERIC,
            research_and_development            NUMERIC,
            selling_general_and_administration  NUMERIC,
            operating_expense                   NUMERIC,
            operating_income                    NUMERIC,
            ebit                                NUMERIC,
            ebitda                              NUMERIC,
            pretax_income                       NUMERIC,
            net_income                          NUMERIC,
            diluted_eps                         NUMERIC,
            basic_eps                           NUMERIC,
            interest_expense                    NUMERIC,

            -- Cash flow statement (6 columns)
            operating_cash_flow                 NUMERIC,
            capital_expenditure                 NUMERIC,
            free_cash_flow                      NUMERIC,
            repurchase_of_capital_stock         NUMERIC,
            cash_dividends_paid                 NUMERIC,
            depreciation_amortization_depletion NUMERIC,

            -- Balance sheet (13 columns)
            total_assets                        NUMERIC,
            total_liabilities                   NUMERIC,
            stockholders_equity                 NUMERIC,
            cash_and_equivalents                NUMERIC,
            long_term_debt                      NUMERIC,
            current_assets                      NUMERIC,
            current_liabilities                 NUMERIC,
            shares_outstanding                  NUMERIC,
            retained_earnings                   NUMERIC,
            net_ppe                             NUMERIC,
            accounts_receivable                 NUMERIC,
            inventory                           NUMERIC,
            total_debt                          NUMERIC,

            period_type                         TEXT NOT NULL,
            updated_at                          TIMESTAMP NOT NULL DEFAULT NOW(),

            PRIMARY KEY (ticker, period_end, period_type)
        );
    """)

    print("Creating index for ticker lookup...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS financial_history_ticker_idx
        ON financial_history (ticker);
    """)

    conn.commit()
    cursor.close()
    conn.close()

    print("financial_history table ready (wide format).")


if __name__ == "__main__":
    init()
