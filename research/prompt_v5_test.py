"""
research/prompt_v5_test.py

v5 candidate: drops Public perception and Market position entirely,
keeping only Core business lines and Industry/theme classification.

Motivation (2026-08-11): today's fabrication audit (v1-v4, 20 tickers)
found 60% of confirmed fabrications in Public perception and 30% in
Market position, vs. 10% in Core business lines and 0% in Industry
classification. Discovery only needs "what does this company do /
what industry is it in" for theme-based retrieval — competitor lookups
should route through stock_universe.peers (FMP data), not through
whatever competitor name a Public perception paragraph happens to
mention. Dropping the two highest-risk, lowest-utility-for-Discovery
dimensions should eliminate most fabrication risk without losing the
retrieval-relevant content.

Retains v3's RETAIN / DO NOT ADD rules unchanged — the issue was
scope (what to write about), not the grounding rules themselves.

Test plan:
  1. Re-run on known-fabrication tickers (CAT, DIS, NFLX, GS, V) —
     confirm the specific fabricated phrases no longer appear.
  2. Re-embed a handful of v5 outputs and re-run similarity search
     against the calibration queries (esp. "semiconductors") to check
     whether contamination cases like Sempra's "next-generation
     technologies" no longer surface.

Throwaway research script — not part of the production pipeline.

Usage:
    python research/prompt_v5_test.py --tickers CAT DIS NFLX GS V
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.update_stock_overviews import fetch_business_text, client
from config import LLM_MODEL

V3_PROMPT = """Based on the following official 10-K business description, \
write a concise company overview (150-200 words) that captures:
- Core business lines (what they actually do)
- Market position (leader, challenger, niche player)
- Public perception (what ordinary people/media commonly associate this company with)
- Industry/theme classification (multiple applicable tags, e.g. technology, \
e-commerce, cloud computing)

Two rules govern every claim in the overview:

1. RETAIN what the text states. If the 10-K text names a specific product, \
brand, technology, strategic initiative, or figure that is relevant to the \
company's business or identity, include it — even if it appears only once, \
appears in a long list alongside other items, or appears late in a long \
filing. Do not omit a real, named detail just to keep the overview short; \
prioritize which details to keep, but do not silently drop ones that are \
clearly material to the company's business (flagship products, named \
strategic principles, core brands).

2. DO NOT ADD what the text does not state. Never supply a specific number, \
percentage, ranking, or comparison that is not written in the text below, \
even if you believe you know the correct or current figure from other \
sources. If the text doesn't give a number, describe the fact qualitatively \
instead of inventing or recalling one.

10-K Business Description:
{business_text}

Write the overview now, in plain, natural language (not bullet points):"""

V5_PROMPT = """Based on the following official 10-K business description, \
write a concise company overview (100-150 words) that captures ONLY:
- Core business lines: what the company actually does — its products, \
services, and operating segments, in concrete terms.
- Industry/theme classification: the specific industry and theme tags this \
company belongs to (e.g. technology, semiconductors, e-commerce, cloud \
computing, healthcare, energy) — as many applicable tags as the text supports.

Do NOT include the company's competitive position, market ranking, leadership \
claims, public/media perception, brand sentiment, or any evaluative language \
about how the company is regarded. This overview exists purely to support \
industry-based search — describe what the company does and what industry it \
is in, nothing else.

Two rules govern every claim in the overview:

1. RETAIN what the text states. If the 10-K text names a specific product, \
brand, technology, or business segment relevant to what the company does, \
include it — even if it appears only once, appears in a long list, or \
appears late in a long filing. Do not omit a real, named detail just to keep \
the overview short.

2. DO NOT ADD what the text does not state. Never supply a specific number, \
percentage, ranking, or comparison that is not written in the text below, \
even if you believe you know it from other sources.

10-K Business Description:
{business_text}

Write the overview now, in plain, natural language (not bullet points):"""


def generate(prompt_template: str, business_text: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt_template.format(business_text=business_text)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"\n{'=' * 78}")
        print(f"{ticker}")
        print("=" * 78)

        business_text, filing_date = fetch_business_text(ticker, None)
        if business_text is None:
            print(f"  Could not fetch Item 1 for {ticker} — skipping")
            continue

        v3_overview = generate(V3_PROMPT, business_text)
        v5_overview = generate(V5_PROMPT, business_text)

        print(f"\n--- v3 (filing {filing_date}) ---")
        print(v3_overview)
        print(f"\n--- v5 (filing {filing_date}) ---")
        print(v5_overview)


if __name__ == "__main__":
    main()
