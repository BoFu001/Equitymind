"""
research/prompt_ab_test.py

A/B test: same 10-K Item 1 text, old OVERVIEW_PROMPT vs a candidate
with a broadened no-outside-knowledge constraint. Checks whether the
fix removes NVDA's suspected training-data leak ("78% of the world's
supercomputers") without degrading the other four tickers, where the
original overview was already accurate and grounded.

Root cause found by manual fact-checking five overviews (2026-08-11):
the constraint sentence in OVERVIEW_PROMPT only names "Public
perception" — it does not cover "Market position", the dimension NVDA's
suspect claim actually falls under. NVIDIA's TOP500 share was 81% as of
June 2026; the overview said "over 78%", which matches the June 2025
figure (77%) more closely than the current one — consistent with the
model recalling a stale number from training data rather than reading
it from Item 1 (10-K Item 1 does not typically report volatile
industry-ranking statistics like this).

Throwaway research script — not part of the production pipeline.

Usage:
    python research/prompt_ab_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.update_stock_overviews import fetch_business_text, client
from config import LLM_MODEL

# The current production prompt (scripts/update_stock_overviews.py).
OLD_PROMPT = """Based on the following official 10-K business description, \
write a concise company overview (150-200 words) that captures:
- Core business lines (what they actually do)
- Market position (leader, challenger, niche player)
- Public perception (what ordinary people/media commonly associate this company with)
- Industry/theme classification (multiple applicable tags, e.g. technology, \
e-commerce, cloud computing)

Public perception must be grounded in brands, products, or programs explicitly \
named in the 10-K text below. Do not introduce outside knowledge.

10-K Business Description:
{business_text}

Write the overview now, in plain, natural language (not bullet points):"""

# Candidate fix: the no-outside-knowledge constraint now covers all
# four dimensions, not just Public perception, and explicitly forbids
# supplying a statistic or ranking the source text does not state.
NEW_PROMPT = """Based on the following official 10-K business description, \
write a concise company overview (150-200 words) that captures:
- Core business lines (what they actually do)
- Market position (leader, challenger, niche player)
- Public perception (what ordinary people/media commonly associate this company with)
- Industry/theme classification (multiple applicable tags, e.g. technology, \
e-commerce, cloud computing)

Every claim above — market position, public perception, and everything else —
must be grounded explicitly in the 10-K text below. Do not introduce facts,
figures, comparisons, rankings, or statistics from outside knowledge, even if
commonly believed to be true. If the text does not state a specific number or
ranking, do not supply one from memory or omit that detail entirely.

10-K Business Description:
{business_text}

Write the overview now, in plain, natural language (not bullet points):"""

TEST_TICKERS = ["NVDA", "JPM", "XOM", "AMZN", "ISRG"]


def generate(prompt_template: str, business_text: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt_template.format(business_text=business_text)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    for ticker in TEST_TICKERS:
        print(f"\n{'=' * 78}")
        print(f"{ticker}")
        print("=" * 78)

        # stored_filing_date=None forces a fresh fetch regardless of
        # what is already in stock_universe — this is a comparison of
        # prompts, not a check of whether the stored overview is current.
        business_text, filing_date = fetch_business_text(ticker, None)
        if business_text is None:
            print(f"  Could not fetch Item 1 for {ticker} — skipping")
            continue

        old_overview = generate(OLD_PROMPT, business_text)
        new_overview = generate(NEW_PROMPT, business_text)

        print(f"\n--- OLD PROMPT (filing {filing_date}) ---")
        print(old_overview)
        print(f"\n--- NEW PROMPT (filing {filing_date}) ---")
        print(new_overview)

        if old_overview.strip() == new_overview.strip():
            print("\n  [identical output]")


if __name__ == "__main__":
    main()
