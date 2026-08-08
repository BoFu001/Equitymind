"""
src/readers/momentum_reader.py

Reads precomputed momentum percentiles for one ticker from the
momentum_benchmarks table (written by update_momentum_benchmarks.py).

Percentile ranking requires the full universe computed together (see
update_momentum_benchmarks.py's docstring), so this reader — like
every other reader in this project — only fetches; it does not
compute anything itself.
"""

import psycopg2
from config import DATABASE_URL



def get_momentum_inputs(ticker: str) -> dict | None:
    """
    Fetches this ticker's precomputed momentum values from
    momentum_benchmarks. Returns None if the ticker has no row (e.g.
    insufficient price history at the last batch run).

    Returns:
        dict with keys: momentum_12_1_pct, momentum_12_1_percentile,
        position_52w, position_52w_percentile
        or None if not found.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT momentum_12_1_pct, momentum_12_1_percentile, "
        "position_52w, position_52w_percentile "
        "FROM momentum_benchmarks WHERE ticker = %s",
        (ticker,),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        return None

    return {
        "momentum_12_1_pct":        row[0],
        "momentum_12_1_percentile": row[1],
        "position_52w":             row[2],
        "position_52w_percentile":  row[3],
    }
