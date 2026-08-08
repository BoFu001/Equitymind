"""
scripts/init_db_sec_chunks.py

One-time database setup script for EquityMind pgvector store.

Run once to:
1. Enable pgvector extension
2. Create sec_chunks table

Usage:
    python scripts/init_db_sec_chunks.py
"""


import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL



def init():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("Enabling pgvector extension...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    print("Creating sec_chunks table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_chunks (
            id              SERIAL PRIMARY KEY,
            ticker          TEXT NOT NULL,
            filing_type     TEXT NOT NULL,
            filing_date     TEXT NOT NULL,
            section         TEXT NOT NULL,
            section_label   TEXT NOT NULL,
            text            TEXT NOT NULL,
            embedding       vector(1536) NOT NULL
        );
    """)

    print("Creating index for vector similarity search...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS sec_chunks_embedding_idx
        ON sec_chunks
        USING hnsw (embedding vector_cosine_ops);
    """)

    print("Creating index for ticker lookup...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS sec_chunks_ticker_idx
        ON sec_chunks (ticker);
    """)

    conn.commit()
    cursor.close()
    conn.close()

    print("Database initialised. Ready.")


if __name__ == "__main__":
    init()