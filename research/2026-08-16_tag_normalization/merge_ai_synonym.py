"""
research/2026-08-16_tag_normalization/merge_ai_synonym.py

Merges "ai" and "artificial intelligence" — a synonym pair missed by
the original plural-only merge (merge_plural_tags.py only checked for
singular/plural variants, not abbreviation/full-form pairs). Found via
test_top_tags.py: querying "ai companies" round-tripped to "artificial
intelligence" instead of "ai", and the two tags turned out to have
almost no ticker overlap (only SNOW in both) — meaning this wasn't a
harmless spelling difference like the chromatography case, but a real
split that would silently omit real AI companies (NVDA, MSFT, GOOGL,
etc., all only tagged "ai") from any query resolving to "artificial
intelligence" instead.

Standard form chosen: "artificial intelligence" (the fuller, less
ambiguous spelling).
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "merged_tag_companies.json"

with open(DATA_PATH) as f:
    data = json.load(f)

ai_entry = None
full_entry = None
other_entries = []

for item in data:
    if item["tag"] == "ai":
        ai_entry = item
    elif item["tag"] == "artificial intelligence":
        full_entry = item
    else:
        other_entries.append(item)

merged_tickers = sorted(set(ai_entry["tickers"]) | set(full_entry["tickers"]))
merged_entry = {
    "tag": "artificial intelligence",
    "count": len(merged_tickers),
    "tickers": merged_tickers,
}

print(f"'ai': {ai_entry['tickers']}")
print(f"'artificial intelligence': {full_entry['tickers']}")
print(f"merged: {merged_tickers} ({len(merged_tickers)} companies)")

new_data = other_entries + [merged_entry]
new_data.sort(key=lambda x: -x["count"])

with open(DATA_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"\nWritten back to {DATA_PATH}")
print(f"Total unique tags now: {len(new_data)}")
