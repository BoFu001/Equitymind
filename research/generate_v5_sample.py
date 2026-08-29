"""
research/generate_v5_sample.py

Generates v5-prompt overviews and embeddings for a sample of tickers,
without touching production stock_universe data. The sample is the
union of all tickers appearing in the v3 baseline's Top-20 rankings
across the six calibration queries (research/v3_threshold_baseline.json),
plus SRE (Sempra) explicitly — the confirmed contamination case from
today's semiconductors-query analysis.

Output: research/v5_sample_overviews.json — {ticker: {overview_text,
embedding}}, used by the next script to re-run the six calibration
queries against v5 embeddings and compare rankings against the saved
v3 baseline.

Throwaway research script — not part of the production pipeline.

Usage:
    python research/generate_v5_sample.py
"""

import json
import sys
import time
from pathlib import Path

import psycopg2
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.update_stock_overviews import fetch_business_text
from src.sec_pipeline.embedder import EMBEDDING_MODEL
from config import DATABASE_URL, OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

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


def load_sample_tickers() -> list[str]:
    baseline_path = Path(__file__).parent / "v3_threshold_baseline.json"
    data = json.loads(baseline_path.read_text())
    tickers = set()
    for query_data in data["queries"].values():
        for row in query_data["top20"]:
            tickers.add(row["ticker"])
    tickers.add("SRE")  # explicit contamination case, may not be in any top-20
    return sorted(tickers)


def generate_v5_overview(business_text: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": V5_PROMPT.format(business_text=business_text)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def embed_overview(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def main():
    tickers = load_sample_tickers()
    print(f"Sample size: {len(tickers)} tickers (union of six queries' Top-20 + SRE)\n")

    results = {}
    failed = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", flush=True)

        business_text, filing_date = fetch_business_text(ticker, None)
        time.sleep(0.5)

        if business_text is None:
            print(f"    Skipping {ticker}: no usable Item 1")
            failed.append(ticker)
            continue

        try:
            overview = generate_v5_overview(business_text)
            embedding = embed_overview(overview)
        except Exception as e:
            print(f"    Skipping {ticker}: generation failed — {e}")
            failed.append(ticker)
            continue

        results[ticker] = {"overview_text": overview, "embedding": embedding, "filing_date": filing_date}

    out_path = Path(__file__).parent / "v5_sample_overviews.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n✓ Generated {len(results)} v5 overviews, saved to {out_path}")
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
