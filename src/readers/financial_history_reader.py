"""
src/readers/financial_history_reader.py

Reads the FULL set of financial_history rows (all periods, all 26
metrics) for a ticker — used when the user's question asks about
historical/multi-year trends (e.g. "Apple's revenue over the last
few years"), not by any single signal's calculation.

This is intentionally a separate file from quality_reader.py
(get_quality_inputs_from_db) — that function serves ONE specific
consumer (quality_signal(), 9 fields, latest 2 annual periods only);
this one serves general historical-trend display (26 fields, every
period currently stored, annual + quarterly). Keeping data readers
one-per-consumer, rather than grouped by "which table they query",
avoids a single file accumulating multiple unrelated responsibilities
(2026-07-27 refactor).
"""


import psycopg2
from config import DATABASE_URL


# financial_history's 26 stored metrics, in column order — grouped by
# statement (income statement / cash flow / balance sheet), matching
# the grouping in update_financial_history.py's METRIC_COLUMNS.
_ALL_METRIC_FIELDS = [
    # Income statement — 13 fields
    "total_revenue",
    "cost_of_revenue",
    "gross_profit",
    "research_and_development",
    "selling_general_and_administration",
    "operating_expense",
    "operating_income",
    "ebit",
    "ebitda",
    "pretax_income",
    "net_income",
    "diluted_eps",
    "basic_eps",

    # Cash flow — 5 fields
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "repurchase_of_capital_stock",
    "cash_dividends_paid",

    # Balance sheet — 8 fields
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "cash_and_equivalents",
    "long_term_debt",
    "current_assets",
    "current_liabilities",
    "shares_outstanding",
]


def get_financial_history_rows(ticker: str) -> list[dict]:
    """
    Returns ALL financial_history rows (annual + quarterly) for a
    ticker, as a list of dicts — for format_financial_history() to
    display when a user's question asks about historical trends (e.g.
    "Apple's revenue over the last few years"). Unlike
    get_quality_inputs_from_db() (which only reads the 9 fields
    Quality needs, and only the latest 2 annual periods), this reads
    all 26 metrics and every period currently stored for the ticker.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    columns = ", ".join(_ALL_METRIC_FIELDS)
    cursor.execute(
        f"""
        SELECT period_end, period_type, {columns}
        FROM financial_history
        WHERE ticker = %s
        ORDER BY period_type, period_end
        """,
        (ticker,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for row in rows:
        period_end, period_type = row[0], row[1]
        metric_values = row[2:]
        entry = {"period_end": period_end, "period_type": period_type}
        # PostgreSQL NUMERIC columns come back as Decimal, not float —
        # same conversion reasoning as get_quality_inputs_from_db.
        for field, value in zip(_ALL_METRIC_FIELDS, metric_values):
            entry[field] = float(value) if value is not None else None
        results.append(entry)

    return results
