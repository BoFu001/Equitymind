"""
scripts/update_common_names.py

Uses an LLM to extract each company's short, common news-headline name
(e.g. "Tesla" from "Tesla, Inc.") from company_name, for matching news
articles by common_name instead of full legal name or ticker.

Reads ticker/company_name from stock_universe table, writes common_name
back to the same table. Does not touch market_cap, peers, or updated_at.

Usage:
    python scripts/update_common_names.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from openai import OpenAI
from config import LLM_MODEL
from config import DATABASE_URL, OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


def extract_common_name(company_name: str) -> str:
    """
    Asks an LLM to extract the short, common name a financial news
    headline would use for this company, given its full legal name.
    """
    prompt = f'''The company\'s official full legal name is: "{company_name}"

Extract the short, common name this company is most often called in
English financial news headlines (not the ticker symbol — the common
name people actually use, in English).

Examples:
"Microsoft Corporation" -> "Microsoft"
"Tesla, Inc." -> "Tesla"
"Amazon.com, Inc." -> "Amazon"
"The Coca-Cola Company" -> "Coca-Cola"
"GE Aerospace" -> "GE"

Reply with ONLY the common name itself, no other text.'''

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def update_common_names():
    print("EquityMind — Common Name Updater")
    print("=" * 50)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Only tickers missing a common_name — existing ones don't change
    # (a company's short news-headline name is effectively permanent),
    # so re-running this for the whole universe every day would waste
    # an LLM call per ticker for nothing. Only newly-added tickers (or
    # any that failed last run and are still NULL) need processing.
    cursor.execute("SELECT ticker, company_name FROM stock_universe WHERE common_name IS NULL")
    rows = cursor.fetchall()

    print(f"\nTickers needing a common_name: {len(rows)}")
    if rows:
        print(f"  {[ticker for ticker, _ in rows]}")
    print("Extracting common names via LLM...\n")

    written = 0

    for i, (ticker, company_name) in enumerate(rows):
        if not company_name:
            print(f"  [{ticker}] No company_name available — skipping")
            continue

        common_name = extract_common_name(company_name)
        print(f"  [{ticker}] {company_name!r} -> {common_name!r}")
        cursor.execute(
            "UPDATE stock_universe SET common_name = %s WHERE ticker = %s",
            (common_name, ticker),
        )
        written += 1

        if (i + 1) % 25 == 0:
            print(f"  ...processed {i + 1}/{len(rows)}")
            conn.commit()

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n✓ stock_universe table updated — {written} new common_name(s) written")

if __name__ == "__main__":
    update_common_names()
