"""
research/2026-08-16_tag_normalization/test_stability.py

Research item 1 (Research Log 05, in progress): stability check for
LLM tag selection. Following Research Log 01's ISRG methodology (a
3-run check at temperature=0 to distinguish true non-determinism from
transcription/reading error), this script runs each test query 3
times through select_tag() and reports whether all 3 runs agree.

Reuses select_tag() from test_tag_selection.py without modifying that
file — this is a read-only consumer of the existing prototype.
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_tag_selection import select_tag

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
    results = []
    for i in range(RUNS_PER_QUERY):
        tag = select_tag(query)
        results.append(tag)

    counts = Counter(results)
    is_stable = len(counts) == 1

    return {
        "query": query,
        "results": results,
        "stable": is_stable,
        "unique_answers": len(counts),
    }


if __name__ == "__main__":
    all_results = []
    print(f"Running {RUNS_PER_QUERY}x stability check on {len(QUERIES)} queries...\n")

    for query in QUERIES:
        result = check_stability(query)
        all_results.append(result)

        status = "STABLE" if result["stable"] else "UNSTABLE"
        print(f"[{status}] \"{query}\"")
        for i, tag in enumerate(result["results"], 1):
            print(f"    Run {i}: {tag}")
        print()

    stable_count = sum(1 for r in all_results if r["stable"])
    print("=" * 50)
    print(f"Summary: {stable_count}/{len(QUERIES)} queries stable across {RUNS_PER_QUERY} runs")
    unstable = [r["query"] for r in all_results if not r["stable"]]
    if unstable:
        print(f"Unstable queries: {unstable}")
