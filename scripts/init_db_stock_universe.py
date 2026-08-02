"""
scripts/init_db_stock_universe.py

Creates the stock_universe table in the same PostgreSQL database used
for quant_signals, financial_history, and sec_chunks.

Written by THREE independent scripts, each updating only its own
column(s), at its own cadence:
    - build_stock_universe.py   writes: market_cap, company_name
    - update_common_names.py    writes: common_name
    - update_peer_groups.py     writes: peers

Low-frequency updates (every few months). peers is stored as JSONB
(list of ticker strings), not a separate join table.

Run once to create the table:
    python scripts/init_db_stock_universe.py
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

    print("Creating stock_universe table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_universe (
            ticker          TEXT PRIMARY KEY,

            -- Written by build_stock_universe.py (market-cap ranking pass)
            market_cap      BIGINT,
            company_name    TEXT,

            -- Written by update_common_names.py (LLM-extracted short name)
            common_name     TEXT,

            -- Written by update_peer_groups.py (FMP stock-peers API)
            peers           JSONB,

            updated_at      TIMESTAMP
        );
    """)
    conn.commit()
    print("stock_universe table ready.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    init()
