"""
Feasibility research: can same-template company overviews be separated by theme
via vector similarity?

This is throwaway research code. It does NOT touch Postgres and does NOT modify
any project module. Results are cached to a local JSON file so the expensive
build phase runs only once.

Usage:
    python discovery_embedding_feasibility.py --build    # fetch + generate + embed
    python discovery_embedding_feasibility.py --eval     # run queries, print verdict
    python discovery_embedding_feasibility.py            # both

Pass criteria (fixed in advance, see PASS_* constants):
    1. Top-3 hit      : >= 2 of the top 3 results belong to the target cluster
    2. Separation     : min(target scores) - max(non-target scores) >= 0.05
    3. No catastrophe : no forbidden-cluster ticker appears in the top 5
"""

import argparse
import json
import os
import time

import numpy as np
from dotenv import load_dotenv
from edgar import Company, set_identity
from openai import OpenAI

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overviews_cache.json")
SEC_IDENTITY = os.getenv("SEC_IDENTITY", "Bo Fu bofu001@gmail.com")
SEC_THROTTLE_SECONDS = float(os.getenv("PIPELINE_THROTTLE_SECONDS", "2"))

CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# 5 clusters x 3 tickers. Robotics is the primary target; semiconductor sits
# adjacent to it (tests fine-grained separation); the other three are distant
# (test the lower bound).
CLUSTERS = {
    "robotics": ["ISRG", "ROK", "TER"],
    "semiconductor": ["NVDA", "INTC", "MU"],
    "banking": ["JPM", "BAC", "GS"],
    "energy": ["XOM", "CVX", "COP"],
    "consumer": ["KO", "PG", "MCD"],
}

TICKER_TO_CLUSTER = {
    ticker: cluster for cluster, tickers in CLUSTERS.items() for ticker in tickers
}

# Each query names its target cluster and the cluster that must not surface in
# the top 5. Consumer staples are the sanity-check floor for every query.
QUERIES = [
    {"theme": "robotics", "target": "robotics", "forbidden": "consumer"},
    {"theme": "semiconductor", "target": "semiconductor", "forbidden": "consumer"},
    {"theme": "banking", "target": "banking", "forbidden": "consumer"},
]

PASS_TOP3_MIN_HITS = 2
PASS_MIN_SEPARATION = 0.05
PASS_FORBIDDEN_TOP_N = 5
MIN_BUSINESS_CHARS = 1000

OVERVIEW_PROMPT = """Based on the following official 10-K business description, \
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

EXPANSION_PROMPT = """Rewrite the user's industry or theme keyword as one sentence \
describing a company in that industry, written in the same style as a company \
business overview. Output the sentence only, with no preamble.

Keyword: {theme}"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------------------------------
# Build phase
# --------------------------------------------------------------------------


def fetch_business_text(ticker):
    """Return (business_text, accession_number) from the most recent 10-K."""
    company = Company(ticker)
    filings = company.get_filings(form="10-K")
    latest = filings.latest()
    if latest is None:
        raise ValueError(f"{ticker}: no 10-K found (foreign issuer files 20-F?)")

    tenk = latest.obj()
    # Real attribute name is lowercase `business`, not `Item1`.
    business_text = str(getattr(tenk, "business", "") or "")
    if len(business_text) < MIN_BUSINESS_CHARS:
        raise ValueError(
            f"{ticker}: business section only {len(business_text)} chars, "
            f"below {MIN_BUSINESS_CHARS} - refusing to generate from it"
        )
    return business_text, latest.accession_no


def generate_overview(business_text):
    """Generate a 150-200 word overview from the full 10-K business section."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": OVERVIEW_PROMPT.format(business_text=business_text),
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def embed(text):
    """Return the embedding vector for a single piece of text."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def expand_theme(theme):
    """Rewrite a bare keyword into a full sentence (HyDE-style query expansion)."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": EXPANSION_PROMPT.format(theme=theme)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    else:
        return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def build():
    """Fetch, generate and embed overviews for every ticker not already cached."""
    set_identity(SEC_IDENTITY)
    cache = load_cache()

    all_tickers = [t for tickers in CLUSTERS.values() for t in tickers]
    todo = [t for t in all_tickers if t not in cache]

    if not todo:
        print(f"All {len(all_tickers)} tickers already cached. Nothing to build.")
        return

    print(f"Building {len(todo)} of {len(all_tickers)} tickers...\n")

    for i, ticker in enumerate(todo, start=1):
        try:
            business_text, accession = fetch_business_text(ticker)
            overview = generate_overview(business_text)
            vector = embed(overview)

            cache[ticker] = {
                "cluster": TICKER_TO_CLUSTER[ticker],
                "accession": accession,
                "business_chars": len(business_text),
                "overview": overview,
                "embedding": vector,
            }
            save_cache(cache)  # save after each ticker so a crash loses nothing

            print(
                f"[{i}/{len(todo)}] {ticker:5s} ok  "
                f"10-K {len(business_text):>7,d} chars  "
                f"overview {len(overview.split()):>3d} words"
            )
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {ticker:5s} FAILED: {exc}")

        time.sleep(SEC_THROTTLE_SECONDS)

    print(f"\nCache written to {CACHE_PATH}")


# --------------------------------------------------------------------------
# Evaluation phase
# --------------------------------------------------------------------------


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank(query_vector, cache):
    """Return [(ticker, cluster, score), ...] sorted by descending similarity."""
    scored = [
        (ticker, entry["cluster"], cosine(query_vector, entry["embedding"]))
        for ticker, entry in cache.items()
    ]
    scored.sort(key=lambda row: row[2], reverse=True)
    return scored


def check_criteria(ranked, target, forbidden):
    """Evaluate the three pass criteria against one ranked list."""
    top3_hits = sum(1 for _, cluster, _ in ranked[:3] if cluster == target)
    c1 = top3_hits >= PASS_TOP3_MIN_HITS

    target_scores = [s for _, cluster, s in ranked if cluster == target]
    other_scores = [s for _, cluster, s in ranked if cluster != target]
    separation = min(target_scores) - max(other_scores)
    c2 = separation >= PASS_MIN_SEPARATION

    top_n_clusters = [cluster for _, cluster, _ in ranked[:PASS_FORBIDDEN_TOP_N]]
    c3 = forbidden not in top_n_clusters

    return {
        "top3_hits": top3_hits,
        "separation": separation,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "all_passed": c1 and c2,
    }


def report_one(label, ranked, target, forbidden):
    """Print a ranked list plus its criteria verdict. Returns the result dict."""
    print(f"  {label}")
    for position, (ticker, cluster, score) in enumerate(ranked, start=1):
        marker = " <-- target" if cluster == target else ""
        print(f"    {position:2d}. {ticker:5s} {cluster:15s} {score:.4f}{marker}")

    result = check_criteria(ranked, target, forbidden)
    tick = lambda ok: "PASS" if ok else "FAIL"
    print(
        f"    criteria: "
        f"top3_hits={result['top3_hits']}/3 [{tick(result['c1'])}]  "
        f"separation={result['separation']:+.4f} [{tick(result['c2'])}]  "
        f"no_{forbidden}_in_top{PASS_FORBIDDEN_TOP_N} [{tick(result['c3'])}]"
    )
    print()
    return result


def evaluate():
    cache = load_cache()
    if not cache:
        print(f"No cache at {CACHE_PATH}. Run with --build first.")
        return

    print(f"Evaluating {len(cache)} tickers across {len(CLUSTERS)} clusters.\n")

    summary = {"raw": [], "expanded": []}

    for query in QUERIES:
        theme, target, forbidden = query["theme"], query["target"], query["forbidden"]
        print(f"QUERY: {theme!r}  (target cluster: {target})")
        print("=" * 60)

        raw_ranked = rank(embed(theme), cache)
        summary["raw"].append(report_one("[raw keyword]", raw_ranked, target, forbidden))

        expanded_text = expand_theme(theme)
        print(f"  [expanded] {expanded_text}")
        expanded_ranked = rank(embed(expanded_text), cache)
        summary["expanded"].append(
            report_one("[expanded sentence]", expanded_ranked, target, forbidden)
        )

    # ---- overall verdict -------------------------------------------------
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    for mode in ("raw", "expanded"):
        results = summary[mode]
        passed = sum(1 for r in results if r["all_passed"])
        seps = [r["separation"] for r in results]
        print(
            f"  {mode:18s} {passed}/{len(results)} queries fully passed   "
            f"separation min={min(seps):+.4f} mean={sum(seps) / len(seps):+.4f}"
        )

    raw_ok = all(r["all_passed"] for r in summary["raw"])
    exp_ok = all(r["all_passed"] for r in summary["expanded"])

    print()
    if raw_ok:
        print("  -> Bare keywords are sufficient. Proceed to the full 250-ticker run.")
    elif exp_ok:
        print("  -> Query expansion is REQUIRED. Proceed, but embed expanded")
        print("     sentences at query time rather than the raw keyword.")
    else:
        weak_c1 = any(not r["c1"] for r in summary["expanded"])
        if weak_c1:
            print("  -> Ranking itself is broken (criterion 1 failed). The approach")
            print("     does not hold. Stop and reconsider before writing more code.")
        else:
            print("  -> Ranking works but separation is too tight. Fixable: raise")
            print("     industry-term density in the overview prompt, or retrieve")
            print("     top-N without a threshold and lean harder on SQL filtering.")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="fetch, generate, embed")
    parser.add_argument("--eval", action="store_true", help="run queries and report")
    args = parser.parse_args()

    run_build = args.build or not (args.build or args.eval)
    run_eval = args.eval or not (args.build or args.eval)

    if run_build:
        build()
    if run_eval:
        if run_build:
            print()
        evaluate()
