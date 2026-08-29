"""
research/2026-08-16_tag_normalization/merge_plural_tags.py

Merges known singular/plural tag pairs in industry_tags_comparison.json
into a single canonical (singular) form, then outputs each merged tag
with the exact list of tickers it applies to.

Background: llm_tags are extracted independently per company (one LLM
call per 10-K), with no cross-company consistency step, so the same
concept can surface under different spellings depending on how each
company's own filing phrases it (e.g. "we operate in the semiconductors
industry" vs "the semiconductor capital equipment sector"). This
produces tag fragmentation — verified pairs found in the 268-tag set:
  - semiconductor (4) / semiconductors (10)   -> 14 combined
  - pharmaceutical (1) / pharmaceuticals (13) -> 14 combined
  - medical device (1) / medical devices (5)  -> 6 combined
  - data center (1) / data centers (1)        -> 2 combined
  - home fashion (1) / home fashions (1)      -> 2 combined

These 5 pairs were found by checking, for every unique tag, whether
adding/removing a trailing 's' produces another tag that also exists
in the set. This is a manual, verified list — not an automatic rule
applied at runtime, because blindly stripping trailing 's' would
incorrectly merge unrelated words (e.g. "gas" and "ga").
"""

import json
from collections import defaultdict
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent.parent / "research" / "industry_tags_comparison.json"

PLURAL_TO_SINGULAR = {
    "semiconductors": "semiconductor",
    "pharmaceuticals": "pharmaceutical",
    "medical devices": "medical device",
    "data centers": "data center",
    "home fashions": "home fashion",
}


def load_data(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def merge_tags(data: list[dict]) -> dict[str, list[str]]:
    """
    Returns {merged_tag: [tickers]}, sorted by ticker count descending.
    """
    tag_to_tickers = defaultdict(list)
    for item in data:
        for tag in item["llm_tags"]:
            merged_tag = PLURAL_TO_SINGULAR.get(tag, tag)
            tag_to_tickers[merged_tag].append(item["ticker"])
    return tag_to_tickers


def main():
    data = load_data(INPUT_PATH)
    print(f"Loaded {len(data)} companies from {INPUT_PATH.name}")

    original_unique = len(set(tag for item in data for tag in item["llm_tags"]))
    tag_to_tickers = merge_tags(data)
    merged_unique = len(tag_to_tickers)

    print(f"Unique tags before merge: {original_unique}")
    print(f"Unique tags after merge:  {merged_unique}")
    print(f"Merged pairs applied: {len(PLURAL_TO_SINGULAR)}")
    print()

    sorted_tags = sorted(tag_to_tickers.items(), key=lambda x: -len(x[1]))

    output_path = Path(__file__).parent / "merged_tag_companies.json"
    with open(output_path, "w") as f:
        json.dump(
            [{"tag": tag, "count": len(tickers), "tickers": tickers} for tag, tickers in sorted_tags],
            f,
            indent=2,
        )
    print(f"Written: {output_path}")

    print()
    print("Preview (top 20 by company count):")
    for tag, tickers in sorted_tags[:20]:
        print(f"  {len(tickers):3d}  {tag}")


if __name__ == "__main__":
    main()
