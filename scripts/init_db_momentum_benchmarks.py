"""
scripts/init_db_momentum_benchmarks.py

Creates the momentum_benchmarks table — replaces momentum_benchmarks.json.
Written by update_momentum_benchmarks.py, read by momentum_signal.py.

Run once to create the table:
    python scripts/init_db_momentum_benchmarks.py
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

    print("Creating momentum_benchmarks table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS momentum_benchmarks (
            ticker                      TEXT PRIMARY KEY,
            momentum_12_1_pct           REAL,
            momentum_12_1_percentile    REAL,
            position_52w                REAL,
            position_52w_percentile     REAL,
            updated_at                  TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)
    conn.commit()
    print("momentum_benchmarks table ready.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    init()
