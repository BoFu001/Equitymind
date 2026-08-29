"""
research/2026-08-16_synonym_merge/test_merge_impact.py

Simulates applying candidate_synonyms.py's merges to the current
merged_tag_companies.json, WITHOUT writing anything back — just
reports the before/after picture, specifically focused on whether
this reduces the long-tail (single-company) problem.
"""

import json
from pathlib import Path
from collections import defaultdict

from candidate_synonyms import CANDIDATE_SYNONYMS

DATA_PATH = Path(__file__).parent.parent / "2026-08-16_tag_normalization" / "merged_tag_companies.json"

with open(DATA_PATH) as f:
    data = json.load(f)

# Before: count single-company (long-tail) tags
before_single = [item["tag"] for item in data if item["count"] == 1]
print(f"BEFORE merge: {len(data)} total tags, {len(before_single)} single-company tags")

# Apply the candidate merges
tag_to_tickers = defaultdict(set)
for item in data:
    canonical = CANDIDATE_SYNONYMS.get(item["tag"], item["tag"])
    tag_to_tickers[canonical].update(item["tickers"])

after_data = [{"tag": tag, "count": len(tickers), "tickers": sorted(tickers)} for tag, tickers in tag_to_tickers.items()]
after_single = [item["tag"] for item in after_data if item["count"] == 1]

print(f"AFTER merge:  {len(after_data)} total tags, {len(after_single)} single-company tags")
print()

print("Details of each candidate merge:")
for old_tag, canonical in CANDIDATE_SYNONYMS.items():
    old_entry = next((item for item in data if item["tag"] == old_tag), None)
    canonical_entry = next((item for item in data if item["tag"] == canonical), None)
    old_tickers = old_entry["tickers"] if old_entry else []
    canonical_tickers = canonical_entry["tickers"] if canonical_entry else []
    merged_tickers = sorted(set(old_tickers) | set(canonical_tickers))
    print(f"  '{old_tag}' ({len(old_tickers)}) + '{canonical}' ({len(canonical_tickers)}) -> {len(merged_tickers)} companies: {merged_tickers}")
