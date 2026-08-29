"""
research/2026-08-16_tag_normalization/compare_methods.py

Runs the same set of industry queries through both retrieval methods
side by side:
  1. Whole-overview embedding similarity (current production method,
     get_industry_tickers in discovery_experiment.py) — shows Top 20
     by cosine similarity.
  2. LLM tag selection against the merged tag vocabulary
     (merged_tag_companies.json) — shows the exact ticker set for the
     selected tag.

This does NOT modify discovery_experiment.py — it reimplements the
embedding-similarity logic locally (same query, same DB table, same
formula) so both methods can be run and compared in one script without
touching production code.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import psycopg2
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, DATABASE_URL
from src.sec_pipeline.embedder import EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = Path(__file__).parent / "merged_tag_companies.json"
with open(DATA_PATH) as f:
    tag_data = json.load(f)
TAG_TO_TICKERS = {item["tag"]: item["tickers"] for item in tag_data}
ALL_TAGS = list(TAG_TO_TICKERS.keys())


def embedding_similarity_top20(query: str) -> list[tuple[str, float]]:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, overview_embedding FROM stock_universe WHERE overview_embedding IS NOT NULL"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    query_vector = np.array(
        client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
    )
    scored = []
    for ticker, embedding in rows:
        if isinstance(embedding, str):
            company_vector = np.array([float(x) for x in embedding.strip("[]").split(",")])
        else:
            company_vector = np.array(embedding, dtype=float)
        similarity = float(
            np.dot(query_vector, company_vector)
            / (np.linalg.norm(query_vector) * np.linalg.norm(company_vector))
        )
        scored.append((ticker, similarity))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:20]


def select_tag(question: str) -> str | None:
    tag_list_str = ", ".join(ALL_TAGS)
    prompt = f"""You are matching a user's industry/theme question to the SINGLE closest tag from a fixed vocabulary.

Available tags (pick exactly one, verbatim, from this list):
{tag_list_str}

User's question: "{question}"

Respond with ONLY the exact tag text from the list above that best matches what the user is asking about. If nothing in the list is a reasonable match, respond with exactly: NONE

Your answer (just the tag, nothing else):"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    if answer == "NONE" or answer not in TAG_TO_TICKERS:
        return None
    return answer


def compare(query: str):
    print(f"{'=' * 70}")
    print(f"QUERY: \"{query}\"")
    print(f"{'=' * 70}")

    print("\n--- Method 1: Embedding similarity (Top 20 of 243, no cutoff) ---")
    top20 = embedding_similarity_top20(query)
    for ticker, score in top20:
        print(f"    {ticker:6s} {score:.4f}")

    print("\n--- Method 2: Tag selection (exact match) ---")
    tag = select_tag(query)
    if tag is None:
        print("    No matching tag found")
        tag_tickers = set()
    else:
        tag_tickers = set(TAG_TO_TICKERS[tag])
        print(f"    Selected tag: '{tag}' ({len(tag_tickers)} companies)")
        print(f"    Tickers: {sorted(tag_tickers)}")

    embedding_top20_tickers = set(t for t, s in top20)
    overlap = embedding_top20_tickers & tag_tickers
    only_in_embedding_top20 = embedding_top20_tickers - tag_tickers
    only_in_tag = tag_tickers - embedding_top20_tickers

    print("\n--- Comparison ---")
    print(f"    In both (tag result AND embedding Top 20): {sorted(overlap)}")
    print(f"    In embedding Top 20 but NOT in tag result:  {sorted(only_in_embedding_top20)}")
    print(f"    In tag result but NOT in embedding Top 20:  {sorted(only_in_tag)}")
    print()


if __name__ == "__main__":
    queries = [
        "semiconductor stocks",
        "healthcare companies",
        "chip stocks",
        "IT companies",
        "chromatography equipment",
    ]
    for q in queries:
        compare(q)
