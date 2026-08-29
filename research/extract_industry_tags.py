"""
research/extract_industry_tags.py

Extracts industry/theme tags from all 243 generated company overviews,
using two independent methods for cross-validation:

1. Regex extraction (baseline, already run manually — see conversation
   log 2026-08-11). Matches template phrases like "classified under...
   including X, Y, Z". Known limitation: brittle to wording variation,
   caught template noise ("several industry themes" counted as a tag),
   and has unknown recall — no way to confirm it caught every overview's
   tags without independently checking.

2. LLM extraction (this script). Asks the model to list only the
   industry/theme tags explicitly stated in the overview's final
   sentence(s), nothing more, nothing less. Uses LLM_MODEL_LIGHT since
   this is a pure extraction task (no synthesis, no judgment call about
   relevance) — the failure mode to guard against is over-generation
   (inventing tags), not model capability.

This produces two frequency tables for comparison, plus a sample for
manual spot-checking against the source overview text — the same
"verify against ground truth, don't trust either method blindly"
methodology used throughout today's prompt calibration research.

Throwaway research script — not part of the production pipeline.

Usage:
    python research/extract_industry_tags.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from openai import OpenAI

from config import DATABASE_URL, LLM_MODEL_LIGHT, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

EXTRACTION_PROMPT = """The following is a company business overview. It ends \
with a sentence or two naming the industry/theme tags this company is \
classified under (phrases like "classified under X, Y, Z" or "industry \
themes including X, Y, Z" or "operating within the X sector").

Extract ONLY the specific industry/theme tags explicitly named in that \
sentence. Do not include generic phrases like "several industry themes" or \
"multiple sectors" — those are not tags themselves, they are the sentence's \
introductory phrase. Do not add any tag that is not explicitly named. Do not \
omit any tag that is named, even if it seems redundant with another.

Return ONLY a JSON array of strings, nothing else. If no tags can be \
identified, return an empty array.

Overview:
{overview_text}

JSON array:"""


def extract_tags_llm(overview_text: str) -> list[str]:
    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(overview_text=overview_text)}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if the model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        tags = json.loads(raw)
        if isinstance(tags, list):
            return [str(t).strip().lower() for t in tags if str(t).strip()]
        else:
            return []
    except json.JSONDecodeError:
        print(f"    [parse failed, raw output: {raw[:100]!r}]")
        return []


def extract_tags_regex(overview_text: str) -> list[str]:
    """The original baseline method — reproduced here for direct
    side-by-side comparison rather than relying on a memory of its
    earlier ad-hoc run."""
    match = re.search(
        r"(classified under|industry themes?[:,]?\s*including)([^.]*)",
        overview_text,
        re.IGNORECASE,
    )
    if not match:
        return []
    tags_text = match.group(2)
    tags = re.split(r",|\sand\s", tags_text)
    return [t.strip().strip(".").lower() for t in tags if t.strip() and len(t.strip()) < 40]


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, overview_text FROM stock_universe WHERE overview_text IS NOT NULL ORDER BY ticker")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"Extracting tags from {len(rows)} overviews using two methods...\n")

    regex_counter = Counter()
    llm_counter = Counter()
    per_ticker_comparison = []

    for i, (ticker, text) in enumerate(rows, start=1):
        regex_tags = extract_tags_regex(text)
        llm_tags = extract_tags_llm(text)

        regex_counter.update(regex_tags)
        llm_counter.update(llm_tags)

        per_ticker_comparison.append({
            "ticker": ticker,
            "regex_tags": regex_tags,
            "llm_tags": llm_tags,
        })

        if i % 50 == 0:
            print(f"  ...processed {i}/{len(rows)}")

    print(f"\n{'=' * 70}")
    print("REGEX extraction — top 30")
    print("=" * 70)
    for tag, count in regex_counter.most_common(30):
        print(f"  {count:3d}  {tag}")

    print(f"\n{'=' * 70}")
    print("LLM extraction — top 30")
    print("=" * 70)
    for tag, count in llm_counter.most_common(30):
        print(f"  {count:3d}  {tag}")

    # Tags found by one method but not the other at all
    regex_only = set(regex_counter) - set(llm_counter)
    llm_only = set(llm_counter) - set(regex_counter)

    print(f"\n{'=' * 70}")
    print(f"Tags ONLY found by regex ({len(regex_only)}):")
    print("=" * 70)
    for tag in sorted(regex_only, key=lambda t: -regex_counter[t])[:20]:
        print(f"  {regex_counter[tag]:3d}  {tag}")

    print(f"\n{'=' * 70}")
    print(f"Tags ONLY found by LLM ({len(llm_only)}):")
    print("=" * 70)
    for tag in sorted(llm_only, key=lambda t: -llm_counter[t])[:20]:
        print(f"  {llm_counter[tag]:3d}  {tag}")

    # Tickers where regex found nothing (potential recall failure)
    regex_zero = [row["ticker"] for row in per_ticker_comparison if not row["regex_tags"]]
    llm_zero = [row["ticker"] for row in per_ticker_comparison if not row["llm_tags"]]
    print(f"\n{'=' * 70}")
    print(f"Tickers with ZERO tags extracted — regex: {len(regex_zero)}, LLM: {len(llm_zero)}")
    print("=" * 70)
    if regex_zero:
        print(f"  regex missed entirely: {regex_zero[:20]}{' ...' if len(regex_zero) > 20 else ''}")
    if llm_zero:
        print(f"  LLM missed entirely: {llm_zero[:20]}{' ...' if len(llm_zero) > 20 else ''}")

    out_path = Path(__file__).parent / "industry_tags_comparison.json"
    out_path.write_text(json.dumps(per_ticker_comparison, indent=2))
    print(f"\nFull per-ticker comparison saved to {out_path}")


if __name__ == "__main__":
    main()
