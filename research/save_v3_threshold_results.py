"""
research/save_v3_threshold_results.py

Persists the v3-prompt similarity ranking results from today's
threshold calibration run, so the baseline is not lost to terminal
history before the v5 comparison is run.

Six queries selected from the LLM-verified industry tag frequency
table (Research Log 02), spanning the observed density range in the
243-ticker corpus: technology (113 occurrences) down to robotics (3).

Output: research/v3_threshold_baseline.json (full Top-20 rankings,
machine-readable) and research/v3_threshold_baseline.md (human-readable
summary), both to be reused when comparing against v5-prompt overviews.

Throwaway research script — not part of the production pipeline.

Usage:
    python research/save_v3_threshold_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sec_pipeline.embedder import EMBEDDING_MODEL
from config import DATABASE_URL, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

TEST_QUERIES = [
    "technology",
    "financial services",
    "healthcare",
    "pharmaceuticals",
    "semiconductors",
    "robotics",
]


def load_overview_embeddings(cursor):
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


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    universe = load_overview_embeddings(cursor)
    cursor.close()
    conn.close()

    print(f"Loaded {len(universe)} tickers with v3-prompt overview embeddings.")

    results = {"prompt_version": "v3", "universe_size": len(universe), "queries": {}}

    md_lines = [
        "# Discovery Threshold Calibration — v3 Baseline",
        "",
        "Similarity rankings computed against the v3-prompt overview embeddings",
        "(production data as of 2026-08-11, before the v5 comparison).",
        "",
    ]

    for query in TEST_QUERIES:
        query_vec = embed_query(query)
        scored = [
            (ticker, name, cosine_similarity(query_vec, vec))
            for ticker, name, vec in universe
        ]
        scored.sort(key=lambda row: row[2], reverse=True)
        top20 = scored[:20]

        results["queries"][query] = {
            "top1_score": top20[0][2],
            "top20": [
                {"ticker": t, "common_name": n, "score": s}
                for t, n, s in top20
            ],
        }

        print(f"  {query!r}: top-1 = {top20[0][0]} ({top20[0][2]:.4f})")

        md_lines.append(f"## Query: `{query}` (top-1 score: {top20[0][2]:.4f})")
        md_lines.append("")
        md_lines.append("| Rank | Ticker | Company | Score |")
        md_lines.append("|---|---|---|---|")
        for i, (t, n, s) in enumerate(top20, start=1):
            md_lines.append(f"| {i} | {t} | {n} | {s:.4f} |")
        md_lines.append("")

    out_json = Path(__file__).parent / "v3_threshold_baseline.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved machine-readable results to {out_json}")

    out_md = Path(__file__).parent / "v3_threshold_baseline.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved human-readable summary to {out_md}")


if __name__ == "__main__":
    main()
