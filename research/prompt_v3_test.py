"""
research/prompt_v3_test.py

Three-way A/B/C test on prompt versions for update_stock_overviews.py.

v1 = current production prompt (narrow constraint, scoped to Public
     perception only) — baseline, known to leak stale training-data
     figures (NVDA "78%" case).
v2 = broadened constraint tested 2026-08-11 — fixed the leak but
     under-includes real content buried in long lists or long filings
     (dropped AMZN's Alexa/Ring, ISRG's "Quintuple Aim").
v3 = current candidate — splits the constraint into an explicit
     retention duty (don't omit real details for brevity) and a
     fabrication ban (don't add unstated numbers), rather than one
     blanket "no outside knowledge" line. Verified stable across 3
     repeated runs on ISRG (Quintuple Aim kept every time).

Throwaway research script — not part of the production pipeline.

Usage:
    python research/prompt_v3_test.py                       # default set, 1 run each
    python research/prompt_v3_test.py --tickers NVDA JPM XOM # specific tickers
    python research/prompt_v3_test.py --runs 3               # repeat each N times
    python research/prompt_v3_test.py --versions v3          # only run v3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.update_stock_overviews import fetch_business_text, client
from config import LLM_MODEL

V1_PROMPT = """Based on the following official 10-K business description, \
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

V2_PROMPT = """Based on the following official 10-K business description, \
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


V4_PROMPT = """Based on the following official 10-K business description, \
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

A THIRD rule applies specifically to Public perception: this section is the \
highest-risk part of the overview for introducing outside knowledge, because \
it invites you to describe "what people associate this company with" — a \
framing that pulls toward well-known cultural symbols, mascots, slogans, or \
taglines you already know rather than ones stated in the text. Every symbol, \
character, slogan, or association named in Public perception MUST appear \
verbatim in the 10-K text below. If you cannot point to the exact phrase in \
the text, do not include it — describe the company's public identity using \
only the brands, products, and language the filing itself uses.

10-K Business Description:
{business_text}

Write the overview now, in plain, natural language (not bullet points):"""

ALL_PROMPTS = {"v1": V1_PROMPT, "v2": V2_PROMPT, "v3": V3_PROMPT, "v4": V4_PROMPT}

DEFAULT_TICKERS = ["NVDA", "JPM", "XOM", "AMZN", "ISRG"]

# Terms known to be present in each ticker's Item 1 text, checked
# after generation to see which prompt version retains them. Empty
# list means no specific retention check for that ticker — just eyeball
# the output (e.g. NVDA is checked for absence of fabricated stats,
# not presence of a specific term).
KNOWN_PRESENT_TERMS = {
    "AMZN": ["Alexa", "Ring", "Kindle", "Echo"],
    "ISRG": ["Quintuple Aim", "da Vinci", "Ion"],
}


def generate(prompt_template: str, business_text: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt_template.format(business_text=business_text)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--versions", nargs="+", default=list(ALL_PROMPTS.keys()),
                         choices=list(ALL_PROMPTS.keys()))
    parser.add_argument("--runs", type=int, default=1,
                         help="repeat each (ticker, version) this many times, to check stability")
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"\n{'=' * 78}")
        print(f"{ticker}")
        print("=" * 78)

        business_text, filing_date = fetch_business_text(ticker, None)
        if business_text is None:
            print(f"  Could not fetch Item 1 for {ticker} — skipping")
            continue

        print(f"  (filing {filing_date}, {len(business_text):,} chars)\n")

        for version in args.versions:
            prompt = ALL_PROMPTS[version]
            for run_i in range(args.runs):
                run_label = f"{version}" if args.runs == 1 else f"{version} run {run_i + 1}/{args.runs}"
                overview = generate(prompt, business_text)
                print(f"--- {run_label} ---")
                print(overview)

                terms = KNOWN_PRESENT_TERMS.get(ticker, [])
                if terms:
                    checks = [
                        f"{term}={'kept' if term.lower() in overview.lower() else 'DROPPED'}"
                        for term in terms
                    ]
                    print(f"\n  [retention check: {', '.join(checks)}]")
                print()


if __name__ == "__main__":
    main()
