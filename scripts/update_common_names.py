"""
scripts/update_common_names.py

Common Name Updater for EquityMind News Sentiment filtering.

For every ticker in stock_universe.json, uses an LLM to extract the
short, common name a financial news headline would actually use (e.g.
"Tesla" from "Tesla, Inc.", "Coca-Cola" from "The Coca-Cola Company") —
distinct from the full legal company_name already stored there.

This exists because news headlines almost never use a company's full
legal name. Naive string-splitting rules (first word, split on comma,
etc.) were tried first and repeatedly failed on real edge cases:
    - "Microsoft Corporation" -> naive first-word split gives "Microsoft"
      (works), but "Tesla, Inc." -> "Tesla," (comma still attached, fails
      to match "Tesla" in a real headline)
    - "Amazon.com, Inc." -> stripping all punctuation merges "Amazon"
      and "com" into "Amazoncom" (fails to match "Amazon")
    - "The Coca-Cola Company" -> naive first-word split gives "The"
      (a meaningless stopword, fails entirely)
    - "GE Aerospace" -> "GE" (correct, but only 2 characters — collides
      with "GEV" -> "GE Vernova Inc." -> also extracts to "GE")
An LLM call handles all of these correctly because it understands
"The" is an article and "Company"/"Inc." are legal suffixes, not
because it follows a smarter string rule — this is a semantic task,
not a string-parsing task.

Validated (2026-07-18) against real finlight news for 8 companies
(AAPL, MSFT, TSLA, AMZN, XOM, JPM, KO, CLOV) — matching on common_name
alone (not ticker) correctly excluded leveraged/options-income ETF
noise (e.g. "YieldMax TSLA Option Income Strategy ETF..."), which
ticker-based matching had let through. Known limitation: this also
excludes a small number (~5%) of genuinely relevant articles whose
headline uses only the ticker (e.g. "MSFT Stock Has Bounced From This
Price Before") — accepted tradeoff since it's far smaller than the
noise it removes.

Usage:
    python scripts/update_common_names.py

Run whenever stock_universe.json is rebuilt, or if a company_name
formatting issue is discovered (e.g. a company changes its legal name).
"""

import json
from pathlib import Path

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UNIVERSE_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"

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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def update_common_names():
    print("EquityMind — Common Name Updater")
    print(f"Target: {UNIVERSE_PATH}")
    print("=" * 50)

    with open(UNIVERSE_PATH, "r") as f:
        universe = json.load(f)

    details = universe["details"]
    print(f"\nUniverse size: {len(details)} tickers")
    print("Extracting common names via LLM...\n")

    for i, (ticker, info) in enumerate(details.items()):
        company_name = info.get("company_name")
        if not company_name:
            print(f"  [{ticker}] No company_name available — skipping")
            continue

        common_name = extract_common_name(company_name)
        info["common_name"] = common_name

        if (i + 1) % 25 == 0:
            print(f"  ...processed {i + 1}/{len(details)}")

    with open(UNIVERSE_PATH, "w") as f:
        json.dump(universe, f, indent=2)

    print(f"\n✓ stock_universe.json updated with common_name for {len(details)} tickers")


if __name__ == "__main__":
    update_common_names()
