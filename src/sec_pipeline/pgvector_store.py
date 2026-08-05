"""
src/sec_pipeline/pgvector_store.py

PostgreSQL + pgvector vector store for EquityMind.

Functions:
    insert_chunks(chunks)              — batch insert with transaction
    query(embedding, ticker, top_k)    — vector similarity search
"""

import psycopg2
from psycopg2.extras import execute_values
from config import DATABASE_URL, PGVECTOR_BATCH_SIZE
from src.sec_pipeline.sec_types import EmbeddedSecChunk, RetrievedChunk, SecChunk


def get_connection():
    """Get a fresh PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def insert_chunks(chunks: list[EmbeddedSecChunk]) -> None:
    """
    Batch insert chunks into sec_chunks table.
    Uses a single transaction — all chunks inserted or none (rollback on error).
    This guarantees data integrity — no partial data.
    """
    print(f"Inserting {len(chunks)} chunks to pgvector...")

    # Build rows for batch insert — flat structure, no metadata nesting
    rows = []
    for chunk in chunks:
        rows.append((
            chunk["ticker"],
            chunk["filing_type"],
            chunk["filing_date"],
            chunk["section"],
            chunk["section_label"],
            chunk["text"],
            chunk["embedding"],
        ))

    conn   = get_connection()
    cursor = conn.cursor()

    try:
        # Batch insert in chunks of PGVECTOR_BATCH_SIZE
        for i in range(0, len(rows), PGVECTOR_BATCH_SIZE):
            batch = rows[i:i + PGVECTOR_BATCH_SIZE]
            execute_values(
                cursor,
                """
                INSERT INTO sec_chunks
                    (ticker, filing_type, filing_date, section,
                     section_label, text, embedding)
                VALUES %s
                """,
                batch,
                template="(%s, %s, %s, %s, %s, %s, %s::vector)"
            )
            print(f"  {min(i + PGVECTOR_BATCH_SIZE, len(rows))}/{len(rows)} inserted")

        conn.commit()  # ← all chunks committed atomically
        print("Insert complete.")

    except Exception as e:
        conn.rollback()  # ← if anything fails, nothing saved
        print(f"  [pgvector] Upsert failed, rolled back: {e}")
        raise

    finally:
        cursor.close()
        conn.close()


def query(question_embedding: list[float], ticker: str = None, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Find top_k most similar chunks for a given question embedding and ticker.
    Uses cosine similarity (<=> operator from pgvector).
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT text, section, filing_type, filing_date,
               1 - (embedding <=> %s::vector) AS score
        FROM sec_chunks
        WHERE ticker = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (question_embedding, ticker, question_embedding, top_k)
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Return flat dicts — no metadata nesting
    return [
        RetrievedChunk(
            chunk=SecChunk(
                ticker=ticker,
                filing_type=row[2],
                filing_date=row[3],
                section=row[1],
                section_label="",
                text=row[0],
            ),
            score=row[4],
        )
        for row in rows
    ]