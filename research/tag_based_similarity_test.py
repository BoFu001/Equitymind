"""
research/tag_based_similarity_test.py

Tests tag-based similarity retrieval as an alternative to full-overview
similarity for Discovery: instead of embedding the entire 150-200 word
overview (which mixes Core business lines, Market position, Public
perception, and Industry classification), this embeds ONLY the LLM-
extracted industry/theme tags (research/industry_tags_comparison.json,
the llm_tags field — see Research Log 02 for how these were produced
and verified).

Motivation: today's v3-vs-v5 prompt comparison showed that even v5
(which drops Market position/Public perception from the overview
text) only partially resolved contamination cases like SRE (Sempra),
which dropped from rank 19 to rank 21 in the semiconductors query —
still present, not eliminated. Tag-based matching sidesteps this
differently: rather than trying to write a "clean" overview and hope
nothing evaluative leaks in, it retrieves using ONLY the already-
verified tag list, which the underlying extraction (Research Log 02)
showed to be 100% grounded in source text across the tickers spot-
checked (n=3, 16 tags).

Each ticker's tags are joined into a single string (e.g. "semiconductor,
electronics, AI infrastructure") and embedded once — this is Method A
(single merged vector) rather than one embedding per individual tag,
chosen for a first-pass test since it requires no new table, just one
embedding call per ticker.

Uses the SAME six calibration queries as the v3/v5 threshold work, so
results are directly comparable to research/v3_threshold_baseline.json.

Throwaway research script — not part of the production pipeline.
Reads industry_tags_comparison.json (243 tickers, already generated).
Writes tag_embeddings.json (cached, so repeat runs don't re-embed) and
tag_similarity_results.json/.md (the comparison output).

Usage:
    python research/tag_based_similarity_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sec_pipeline.embedder import EMBEDDING_MODEL
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

TEST_QUERIES = [
    "technology",
    "financial services",
    "healthcare",
    "pharmaceuticals",
    "semiconductors",
    "robotics",
]


def load_ticker_tags() -> dict[str, list[str]]:
    path = Path(__file__).parent / "industry_tags_comparison.json"
    data = json.loads(path.read_text())
    return {row["ticker"]: row["llm_tags"] for row in data if row["llm_tags"]}


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_or_load_tag_embeddings(ticker_tags: dict[str, list[str]]) -> dict[str, np.ndarray]:
    cache_path = Path(__file__).parent / "tag_embeddings.json"
    if cache_path.exists():
        print("Loading cached tag embeddings...")
        cached = json.loads(cache_path.read_text())
        return {t: np.array(v) for t, v in cached.items()}

    print(f"Embedding merged tag strings for {len(ticker_tags)} tickers...")
    embeddings = {}
    for i, (ticker, tags) in enumerate(ticker_tags.items(), start=1):
        merged = ", ".join(tags)
        embeddings[ticker] = embed_text(merged)
        if i % 50 == 0:
            print(f"  ...{i}/{len(ticker_tags)}")

    cache_path.write_text(json.dumps(embeddings, indent=2))
    print(f"Saved to {cache_path}")
    return {t: np.array(v) for t, v in embeddings.items()}


def main():
    ticker_tags = load_ticker_tags()
    print(f"Loaded tags for {len(ticker_tags)} tickers (llm_tags, non-empty only).\n")

    tag_embeddings = build_or_load_tag_embeddings(ticker_tags)

    md_lines = [
        "# Tag-Based Similarity Test",
        "",
        "Similarity rankings computed against merged LLM-extracted tag",
        "strings (research/industry_tags_comparison.json, llm_tags field),",
        "compared to the v3 full-overview baseline in v3_threshold_baseline.json.",
        "",
    ]

    v3_baseline_path = Path(__file__).parent / "v3_threshold_baseline.json"
    v3_baseline = json.loads(v3_baseline_path.read_text()) if v3_baseline_path.exists() else None

    for query in TEST_QUERIES:
        qvec = np.array(embed_text(query))
        scored = [
            (ticker, ticker_tags[ticker], cosine_similarity(qvec, vec))
            for ticker, vec in tag_embeddings.items()
        ]
        scored.sort(key=lambda row: row[2], reverse=True)

        print(f"\n{'=' * 78}")
        print(f"QUERY: {query!r}")
        print("=" * 78)
        print(f"\n  Tag-based Top 15:")
        for i, (ticker, tags, score) in enumerate(scored[:15], start=1):
            tag_str = ", ".join(tags[:4])
            print(f"    {i:2d}. {ticker:6s} {score:.4f}   [{tag_str}]")

        if v3_baseline and query in v3_baseline["queries"]:
            v3_top10 = [row["ticker"] for row in v3_baseline["queries"][query]["top20"][:10]]
            tag_top10 = [t for t, _, _ in scored[:10]]
            overlap = set(v3_top10) & set(tag_top10)
            print(f"\n  Overlap with v3 full-overview Top 10: {len(overlap)}/10  {sorted(overlap)}")
            print(f"  v3-only (not in tag-based top10): {sorted(set(v3_top10) - set(tag_top10))}")
            print(f"  tag-only (not in v3 top10): {sorted(set(tag_top10) - set(v3_top10))}")

        md_lines.append(f"## Query: `{query}`")
        md_lines.append("")
        md_lines.append("| Rank | Ticker | Score | Tags |")
        md_lines.append("|---|---|---|---|")
        for i, (ticker, tags, score) in enumerate(scored[:15], start=1):
            md_lines.append(f"| {i} | {ticker} | {score:.4f} | {', '.join(tags[:4])} |")
        md_lines.append("")

    out_md = Path(__file__).parent / "tag_similarity_results.md"
    out_md.write_text("\n".join(md_lines))
    print(f"\n\nSaved full results to {out_md}")


if __name__ == "__main__":
    main()
