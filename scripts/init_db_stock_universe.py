"""
scripts/init_db_stock_universe.py

Creates the stock_universe table in the same PostgreSQL database used
for quant_signals, financial_history, and sec_chunks.

Design: consolidates what used to be two separate JSON files
(stock_universe.json, peer_groups.json) into a single table. This
table is written by THREE independent scripts, each updating only its
own column(s), at its own cadence — not merged into one script:
    - build_stock_universe.py   writes: market_cap, company_name
    - update_common_names.py    writes: common_name
    - update_peer_groups.py     writes: peers

All three run on the same low-frequency cadence (every few months —
this universe's composition, a company's common name, and its peer
identity are all slow-moving facts, unlike the daily-refreshed
quant_signals table), but remain separate scripts so that, e.g.,
re-running update_common_names.py alone never touches peers or
market_cap, and vice versa.

peers is stored as JSONB (a list of ticker strings) rather than a
separate join table — this project has no query pattern today that
needs to search "which tickers list X as a peer" in reverse, so a
normalized many-to-many table would add complexity without current
benefit; revisit if that need arises.

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
