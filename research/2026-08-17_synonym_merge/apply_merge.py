"""
research/2026-08-16_synonym_merge/apply_merge.py

Applies the 5 candidate synonym merges (validated via
test_merge_impact.py) to the current merged_tag_companies.json, and
writes the deduplicated result to a new file in this folder.

Does NOT overwrite the source file
(research/2026-08-16_tag_normalization/merged_tag_companies.json) —
writes to a new, separately-versioned output here, so today's earlier
work stays intact and this can be reviewed before being adopted as
the new canonical vocabulary.
"""

import json
from pathlib import Path
from collections import defaultdict

from candidate_synonyms import CANDIDATE_SYNONYMS

SOURCE_PATH = Path(__file__).parent.parent / "2026-08-16_tag_normalization" / "merged_tag_companies.json"
OUTPUT_PATH = Path(__file__).parent / "merged_tag_companies_v2.json"

with open(SOURCE_PATH) as f:
    data = json.load(f)

tag_to_tickers = defaultdict(set)
for item in data:
    canonical = CANDIDATE_SYNONYMS.get(item["tag"], item["tag"])
    tag_to_tickers[canonical].update(item["tickers"])

new_data = [
    {"tag": tag, "count": len(tickers), "tickers": sorted(tickers)}
    for tag, tickers in tag_to_tickers.items()
]
new_data.sort(key=lambda x: -x["count"])

with open(OUTPUT_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"Source: {len(data)} tags")
print(f"Output: {len(new_data)} tags")
print(f"Written to: {OUTPUT_PATH}")
