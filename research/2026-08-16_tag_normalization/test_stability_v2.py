"""
research/2026-08-16_tag_normalization/test_stability_v2.py

Refinement of test_stability.py: the first version measured whether
the SELECTED TAG STRING was identical across 3 runs. This conflates
two different things — a different tag string can still resolve to
the same ticker set (e.g. "chromatography" and "analytical
instruments" both map to just WAT), which is harmless for the
end user, versus a different tag string resolving to a genuinely
different ticker set, which actually changes what the user sees.

This version measures stability at the RESULT level (ticker set),
not the tag-string level.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_tag_selection import select_tag, TAG_TO_TICKERS

QUERIES = [
    "semiconductor stocks",
    "healthcare companies",
    "financial services companies",
    "renewable energy stocks",
    "aerospace and defense companies",
    "chip stocks",
    "IT companies",
    "chromatography equipment",
    "which phone-selling companies are good",
]

RUNS_PER_QUERY = 3


def check_stability(query: str) -> dict:
    tag_results = []
    ticker_set_results = []

    for i in range(RUNS_PER_QUERY):
        tag = select_tag(query)
        tag_results.append(tag)
        tickers = frozenset(TAG_TO_TICKERS[tag]) if tag else frozenset()
        ticker_set_results.append(tickers)

    tag_stable = len(set(tag_results)) == 1
    result_stable = len(set(ticker_set_results)) == 1

    return {
        "query": query,
        "tags": tag_results,
        "tag_stable": tag_stable,
        "result_stable": result_stable,
    }


if __name__ == "__main__":
    all_results = []
    print(f"Running {RUNS_PER_QUERY}x stability check on {len(QUERIES)} queries...\n")

    for query in QUERIES:
        result = check_stability(query)
        all_results.append(result)

        tag_status = "same tag" if result["tag_stable"] else "DIFFERENT tags"
        result_status = "SAME result" if result["result_stable"] else "DIFFERENT result"
        print(f"\"{query}\"")
        print(f"    Tags across runs: {result['tags']} -> {tag_status}")
        print(f"    Final result: {result_status}")
        print()

    tag_stable_count = sum(1 for r in all_results if r["tag_stable"])
    result_stable_count = sum(1 for r in all_results if r["result_stable"])
    print("=" * 50)
    print(f"Tag-string stable:    {tag_stable_count}/{len(QUERIES)}")
    print(f"Final-result stable:  {result_stable_count}/{len(QUERIES)}")

    truly_unstable = [r["query"] for r in all_results if not r["result_stable"]]
    if truly_unstable:
        print(f"Queries with genuinely different results: {truly_unstable}")
    else:
        print("No query produced a genuinely different result across runs.")
