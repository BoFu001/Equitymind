"""
research/discovery_threshold_calibration.py

Calibrates the relative similarity threshold for Discovery's theme
search against the real 243-ticker overview embedding set, replacing
the 0.55 ratio estimated from a 15-ticker feasibility sample (5
artificially-separated industry clusters).

Test queries are selected from the LLM-verified industry tag frequency
table (see research_log_02_extraction_method_comparison.md), spanning
the real density range found in the 243-ticker corpus: technology
(113 occurrences, densest) down to robotics (3 occurrences, sparsest).
This lets the calibration test whether a single fixed ratio holds
across both crowded and sparse candidate pools, or whether density
requires a dynamic threshold.

Throwaway research script — not part of the production pipeline.

Usage:
    python research/discovery_threshold_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import psycopg2
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sec_pipeline.embedder import EMBEDDING_MODEL
from config import DATABASE_URL, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# Selected from the real, LLM-verified frequency table (Research Log
# 02), spanning the observed density range: 113 down to 3 occurrences
# across 243 overviews.
TEST_QUERIES = [
    "technology",           # 113 — densest
    "financial services",   # 35  — mid-high
    "healthcare",           # 24  — mid
    "pharmaceuticals",      # 13  — mid-low
    "semiconductors",       # 10  — low
    "robotics",             # 3   — sparsest
]

CANDIDATE_RATIOS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def load_overview_embeddings(cursor) -> list[tuple[str, str, np.ndarray]]:
    cursor.execute(
        """
        SELECT ticker, common_name, overview_embedding
        FROM stock_universe
        WHERE overview_embedding IS NOT NULL
        ORDER BY ticker
        """
    )
    rows = cursor.fetchall()
    result = []
    for ticker, common_name, embedding in rows:
        if isinstance(embedding, str):
            vec = np.array([float(x) for x in embedding.strip("[]").split(",")])
        else:
            vec = np.array(embedding, dtype=float)
        result.append((ticker, common_name, vec))
    return result


def embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_all(query_vec: np.ndarray, universe: list[tuple[str, str, np.ndarray]]):
    scored = [
        (ticker, common_name, cosine_similarity(query_vec, vec))
        for ticker, common_name, vec in universe
    ]
    scored.sort(key=lambda row: row[2], reverse=True)
    return scored


def report_query(query: str, ranked: list[tuple[str, str, float]]) -> None:
    top1 = ranked[0][2]

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query!r}   (top-1 score: {top1:.4f})")
    print("=" * 70)

    print("\n  Top 20 by similarity:")
    for i, (ticker, name, score) in enumerate(ranked[:20], start=1):
        gap = "" if i == 1 else f"  (gap: {ranked[i-2][2] - score:+.4f})"
        print(f"    {i:2d}. {ticker:6s} {str(name)[:28]:28s} {score:.4f}{gap}")

    print("\n  Candidates passing each relative threshold (score >= top1 * ratio):")
    for ratio in CANDIDATE_RATIOS:
        cutoff = top1 * ratio
        passing = [row for row in ranked if row[2] >= cutoff]
        tickers = ", ".join(t for t, _, _ in passing[:15])
        more = f" (+{len(passing) - 15} more)" if len(passing) > 15 else ""
        print(f"    ratio={ratio:.2f}  cutoff={cutoff:.4f}  n={len(passing):3d}   {tickers}{more}")


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    universe = load_overview_embeddings(cursor)
    cursor.close()
    conn.close()

    print(f"Loaded {len(universe)} tickers with overview embeddings.")

    for query in TEST_QUERIES:
        query_vec = embed_query(query)
        ranked = rank_all(query_vec, universe)
        report_query(query, ranked)


if __name__ == "__main__":
    main()
