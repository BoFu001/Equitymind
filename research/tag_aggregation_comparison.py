"""
research/tag_aggregation_comparison.py

Compares three tag-similarity aggregation strategies for a single
query, using per-tag embeddings (not merged-tag-string embeddings —
see research/tag_based_similarity_test.py for that earlier method,
which conflated MSFT's five distinct tags into one blended vector and
appeared to suppress its "technology" ranking as a result).

Each ticker's llm_tags (research/industry_tags_comparison.json) are
embedded INDIVIDUALLY here, producing one vector per tag. Three ways
to collapse a ticker's multiple tag-query similarities into one score
are compared:

  - MAX:      the ticker's single highest tag-query similarity.
              Rewards any one precise tag match, regardless of the
              ticker's other tags — closest to "does this company
              touch this domain at all."
  - AVERAGE:  the mean similarity across ALL of the ticker's tags.
              Penalizes tickers with many unrelated tags diluting a
              genuine match — but also penalizes genuinely
              multi-business companies whose other tags are simply
              in different domains.
  - TOP-2 AVERAGE: the mean of the ticker's two highest tag
              similarities. A middle ground — requires more than one
              coincidental tag hit, without averaging across a whole
              tag list of unrelated size.

Motivation: research/tag_based_similarity_test.py (merged-string
method) ranked MSFT outside the top 15 for "technology" despite MSFT's
tags being accurate and directly relevant (['technology', 'cloud
computing', 'ai', 'productivity software', 'gaming']) — the merged
embedding likely blended toward a centroid not closely aligned with
any single tag. This script tests whether keeping tags separate and
choosing an aggregation rule fixes that, and whether it reintroduces
contamination (e.g. VTR/WELL surfacing under "healthcare" via a single
"healthcare" tag despite being real-estate REITs, seen in the
merged-string test).

Throwaway research script — not part of the production pipeline.
Caches per-tag embeddings so repeat runs (different queries) don't
re-embed the same tags twice.

Usage:
    python research/tag_aggregation_comparison.py --query technology
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sec_pipeline.embedder import EMBEDDING_MODEL
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def load_ticker_tags() -> dict[str, list[str]]:
    path = Path(__file__).parent / "industry_tags_comparison.json"
    data = json.loads(path.read_text())
    return {row["ticker"]: row["llm_tags"] for row in data if row["llm_tags"]}


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_or_load_individual_tag_embeddings(ticker_tags: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Returns {unique_tag_text: embedding_vector} — embedding each
    DISTINCT tag string once, not once per ticker (many tickers share
    tags like "technology"), to minimize API calls."""
    cache_path = Path(__file__).parent / "individual_tag_embeddings.json"
    if cache_path.exists():
        print("Loading cached individual tag embeddings...")
        cached = json.loads(cache_path.read_text())
        return {t: np.array(v) for t, v in cached.items()}

    unique_tags = sorted({tag for tags in ticker_tags.values() for tag in tags})
    print(f"Embedding {len(unique_tags)} unique tags (deduplicated across {len(ticker_tags)} tickers)...")

    embeddings = {}
    for i, tag in enumerate(unique_tags, start=1):
        embeddings[tag] = embed_text(tag)
        if i % 50 == 0:
            print(f"  ...{i}/{len(unique_tags)}")

    cache_path.write_text(json.dumps(embeddings, indent=2))
    print(f"Saved to {cache_path}")
    return {t: np.array(v) for t, v in embeddings.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    ticker_tags = load_ticker_tags()
    tag_embeddings = build_or_load_individual_tag_embeddings(ticker_tags)

    qvec = np.array(embed_text(args.query))

    # For each ticker, compute similarity of the query against EACH of
    # its tags individually, then apply all three aggregation rules.
    results = []
    for ticker, tags in ticker_tags.items():
        sims = sorted(
            (cosine_similarity(qvec, tag_embeddings[tag]) for tag in tags if tag in tag_embeddings),
            reverse=True,
        )
        if not sims:
            continue
        max_score = sims[0]
        avg_score = sum(sims) / len(sims)
        top2_score = sum(sims[:2]) / min(2, len(sims))
        results.append({
            "ticker": ticker,
            "tags": tags,
            "max": max_score,
            "avg": avg_score,
            "top2avg": top2_score,
        })

    print(f"\n{'=' * 90}")
    print(f"QUERY: {args.query!r}")
    print("=" * 90)

    for method in ["max", "avg", "top2avg"]:
        ranked = sorted(results, key=lambda r: r[method], reverse=True)
        print(f"\n--- Top 20 by {method.upper()} ---")
        for i, r in enumerate(ranked[:20], start=1):
            tag_str = ", ".join(r["tags"][:4])
            print(f"  {i:2d}. {r['ticker']:6s} {r[method]:.4f}   [{tag_str}]")

    # Save full comparison for later analysis / thesis writeup
    out_path = Path(__file__).parent / f"tag_aggregation_{args.query.replace(' ', '_')}.json"
    out_path.write_text(json.dumps({
        "query": args.query,
        "max_top20": [r["ticker"] for r in sorted(results, key=lambda r: r["max"], reverse=True)[:20]],
        "avg_top20": [r["ticker"] for r in sorted(results, key=lambda r: r["avg"], reverse=True)[:20]],
        "top2avg_top20": [r["ticker"] for r in sorted(results, key=lambda r: r["top2avg"], reverse=True)[:20]],
        "full_results": results,
    }, indent=2))
    print(f"\nSaved full comparison to {out_path}")


if __name__ == "__main__":
    main()
