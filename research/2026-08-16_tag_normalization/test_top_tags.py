"""
research/2026-08-16_tag_normalization/test_top_tags.py

Feeds each of the top-32 tags (by company count) back into select_tag()
as a natural-language query (e.g. "technology" -> "technology
companies"), to see whether the LLM reliably round-trips back to the
same tag it started from, and whether the resulting company count
matches what's in merged_tag_companies.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_tag_selection import select_tag, TAG_TO_TICKERS

TOP_TAGS = [
    "technology", "cloud computing", "financial services", "e-commerce",
    "healthcare", "energy", "insurance", "retail", "pharmaceutical",
    "semiconductor", "utilities", "sustainability", "consumer goods",
    "biotechnology", "renewable energy", "infrastructure", "logistics",
    "real estate", "artificial intelligence", "aerospace", "automotive",
    "manufacturing", "gaming", "cybersecurity", "life sciences",
    "industrial manufacturing", "banking", "investment management",
    "entertainment", "hospitality", "medical device", "telecommunications",
]

if __name__ == "__main__":
    for original_tag in TOP_TAGS:
        query = f"{original_tag} companies"
        expected_count = len(TAG_TO_TICKERS[original_tag])

        selected_tag = select_tag(query)
        if selected_tag is None:
            print(f"{original_tag:25s} (expected {expected_count:3d}) -> NONE")
        else:
            actual_count = len(TAG_TO_TICKERS[selected_tag])
            match = "OK" if selected_tag == original_tag else "DIFFERENT TAG"
            print(f"{original_tag:25s} (expected {expected_count:3d}) -> '{selected_tag}' ({actual_count} companies) [{match}]")
