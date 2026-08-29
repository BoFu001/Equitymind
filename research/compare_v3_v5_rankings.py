"""
research/compare_v3_v5_rankings.py

Compares Discovery similarity rankings between v3-prompt overviews
(production baseline, research/v3_threshold_baseline.json) and
v5-prompt overviews (research/v5_sample_overviews.json, 97-ticker
sample) across the same six calibration queries.

v5 only covers the 97-ticker sample (union of v3's Top-20 across all
six queries, plus SRE). Rankings are therefore computed within this
97-ticker subset for both versions, not the full 243-ticker universe
— this keeps the comparison apples-to-apples: same candidate pool,
different overview text feeding the embedding.

Key check: does SRE (Sempra) still surface in the "semiconductors"
query under v5, given its confirmed v3 contamination was traced to an
evaluative sentence ("next-generation technologies") in the Market
position section that v5 does not generate.

Throwaway research script — not part of the production pipeline.

Usage:
    python research/compare_v3_v5_rankings.py
"""

import json
from pathlib import Path

import numpy as np

TEST_QUERIES = [
    "technology",
    "financial services",
    "healthcare",
    "pharmaceuticals",
    "semiconductors",
    "robotics",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_v3_baseline():
    """Returns {query: [(ticker, name, score), ...]} restricted to the
    97-ticker sample — but since v3's own Top-20 defined that sample,
    all v3 Top-20 tickers are trivially in it. We need v3 scores against
    ALL 97 sample tickers, not just each query's own Top-20, to rank
    fairly within the same pool as v5. This is not available from the
    saved baseline file (it only stored each query's own Top-20), so
    this script re-derives v3-pool coverage from what IS available:
    ticker identity for filtering, not full re-ranking within the
    97-ticker pool. See note in main() about this limitation.
    """
    path = Path(__file__).parent / "v3_threshold_baseline.json"
    return json.loads(path.read_text())


def load_v5_sample():
    path = Path(__file__).parent / "v5_sample_overviews.json"
    data = json.loads(path.read_text())
    return {
        ticker: np.array(entry["embedding"])
        for ticker, entry in data.items()
    }


def embed_query(text: str) -> np.ndarray:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from openai import OpenAI
    from src.sec_pipeline.embedder import EMBEDDING_MODEL
    from config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def main():
    v3_baseline = load_v3_baseline()
    v5_embeddings = load_v5_sample()

    print(f"v5 sample covers {len(v5_embeddings)} tickers.\n")

    for query in TEST_QUERIES:
        query_vec = embed_query(query)

        # v5 ranking, computed fresh within the 97-ticker sample
        v5_scored = [
            (ticker, cosine_similarity(query_vec, vec))
            for ticker, vec in v5_embeddings.items()
        ]
        v5_scored.sort(key=lambda row: row[1], reverse=True)

        # v3 ranking for this query, as saved (already Top-20 within
        # the full 243-ticker universe — not re-restricted to the
        # 97-ticker sample, so this is v3's true Top-20, a superset
        # comparison rather than a pool-matched one. Flagged as a
        # limitation below.)
        v3_top20 = v3_baseline["queries"][query]["top20"]

        print(f"{'=' * 78}")
        print(f"QUERY: {query!r}")
        print("=" * 78)

        print(f"\n  v3 Top 10 (full 243-ticker universe):")
        for i, row in enumerate(v3_top20[:10], start=1):
            print(f"    {i:2d}. {row['ticker']:6s} {row['common_name'][:28]:28s} {row['score']:.4f}")

        print(f"\n  v5 Top 10 (97-ticker sample pool):")
        for i, (ticker, score) in enumerate(v5_scored[:10], start=1):
            print(f"    {i:2d}. {ticker:6s} {score:.4f}")

        # Explicit SRE check for the semiconductors query
        if query == "semiconductors":
            sre_v3_rank = next(
                (i for i, row in enumerate(v3_top20, start=1) if row["ticker"] == "SRE"),
                None,
            )
            sre_v5_rank = next(
                (i for i, (t, s) in enumerate(v5_scored, start=1) if t == "SRE"),
                None,
            )
            sre_v5_score = next((s for t, s in v5_scored if t == "SRE"), None)
            print(f"\n  SRE (Sempra) check:")
            print(f"    v3 rank: {sre_v3_rank if sre_v3_rank else 'not in top 20'}")
            print(f"    v5 rank (within 97-ticker pool): {sre_v5_rank}   score: {sre_v5_score:.4f}" if sre_v5_score else "    v5: SRE not in sample")

        print()


if __name__ == "__main__":
    main()
