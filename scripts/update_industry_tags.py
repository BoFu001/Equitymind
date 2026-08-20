"""
scripts/update_industry_tags.py

Extracts industry/theme tags from each ticker's overview_text and writes
them to stock_universe.llm_tags. Discovery resolves a user's open-ended
theme ("robotics", "banking") to a candidate pool by selecting a tag from
this vocabulary and looking up its ticker list — an exact lookup, not a
similarity ranking (Research Log 05).

Freshness is decided from the database, not from EDGAR: a ticker is
re-tagged only when tags_filing_date differs from overview_filing_date.
That makes this script free to run daily (one query, no API calls, when
nothing has changed) and self-healing — if a run fails partway, or the
OpenAI account runs dry between this script and update_stock_overviews.py,
the two dates stay divergent and the next run picks the ticker up. Asking
EDGAR instead would give the wrong answer in exactly that case: EDGAR
would agree with the already-updated overview_filing_date and the stale
tags would never be noticed.

Tickers with a NULL overview_text (no usable 10-K Item 1 — see
update_stock_overviews.py) are skipped and keep NULL tags, which makes
them invisible to every tag-based query with no special-case code.

The extraction prompt is a closed-set task: restate the tags named in the
overview's classification sentence, nothing more. Research Log 02 measured
this at 16/16 accurate with zero fabrications, against a regex baseline
that silently missed 32% of the corpus.

Usage:
    python scripts/update_industry_tags.py
    python scripts/update_industry_tags.py --all      # re-tag everything
"""

import re
import sys
import json
import argparse
from pathlib import Path

import psycopg2
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL, OPENAI_API_KEY, LLM_MODEL_LIGHT

client = OpenAI(api_key=OPENAI_API_KEY)

# LLM_MODEL_LIGHT, not LLM_MODEL: extraction is a closed-set restatement
# task, so the failure mode to guard against is over-generation, not
# insufficient capability (Research Log 02 2.2).
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


def extract_tags(overview_text: str) -> list[str] | None:
    """Returns the tag list, or None if extraction failed — None and []
    must stay distinguishable: [] means the model read the overview and
    found no tags, None means we never got a usable answer, and only the
    latter should leave the row unwritten so the next run retries it."""
    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(overview_text=overview_text),
        }],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        tags = json.loads(raw)
    except json.JSONDecodeError:
        print(f"    parse failed, raw output: {raw[:100]!r}", flush=True)
        return None

    if isinstance(tags, list):
        return [str(t).strip().lower() for t in tags if str(t).strip()]
    else:
        print(f"    unexpected JSON type: {type(tags).__name__}", flush=True)
        return None


def _load_targets(cursor, regenerate_all: bool) -> list[tuple[str, str, str, list | None]]:
    """Returns (ticker, overview_text, overview_filing_date, existing_tags).

    The existing tags are carried along so a re-tag can be diffed against
    what it replaces — see the change report in update_industry_tags()."""
    if regenerate_all:
        cursor.execute("""
            SELECT ticker, overview_text, overview_filing_date, llm_tags
            FROM stock_universe
            WHERE overview_text IS NOT NULL
            ORDER BY ticker
        """)
    else:
        cursor.execute("""
            SELECT ticker, overview_text, overview_filing_date, llm_tags
            FROM stock_universe
            WHERE overview_text IS NOT NULL
              AND (tags_filing_date IS DISTINCT FROM overview_filing_date)
            ORDER BY ticker
        """)
    return cursor.fetchall()


def update_industry_tags(regenerate_all: bool = False):
    print("Connecting to PostgreSQL...", flush=True)
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    targets = _load_targets(cursor, regenerate_all)
    print(f"{len(targets)} ticker(s) need tag extraction", flush=True)

    written = 0
    failed = []
    changed = []

    for i, (ticker, overview_text, filing_date, old_tags) in enumerate(targets, 1):
        print(f"  [{i}/{len(targets)}] {ticker}...", flush=True)

        tags = extract_tags(overview_text)
        if tags is None:
            failed.append(ticker)
            continue

        cursor.execute(
            """
            UPDATE stock_universe
            SET llm_tags = %s,
                tags_filing_date = %s
            WHERE ticker = %s
            """,
            (json.dumps(tags), filing_date, ticker),
        )
        conn.commit()  # per ticker, so an interrupted run keeps its work
        written += 1
        print(f"    {tags}", flush=True)

        # A company that already had tags and now has different ones is
        # worth seeing. Tags are re-derived whenever its overview is
        # regenerated, and the overview is regenerated from the same 10-K
        # whenever the row is rebuilt — so a company can silently lose a
        # tag central to its business without its filing having changed.
        # AVGO dropped "semiconductors" for "telecommunications" this way
        # on 2026-08-18, while its own overview text still called it a
        # prominent player in the semiconductor industry four times over.
        # Nothing in the run's output showed it. This diff is the signal.
        if old_tags is not None:
            lost = sorted(set(old_tags) - set(tags))
            gained = sorted(set(tags) - set(old_tags))
            if lost or gained:
                changed.append(ticker)
                print(f"    CHANGED — lost {lost}, gained {gained}", flush=True)

    print(f"\n✓ stock_universe table updated — {written} ticker(s) tagged",
          flush=True)
    if changed:
        print(f"  Tags CHANGED for {len(changed)}: {', '.join(changed)}",
              flush=True)
        print("    (re-derived from a regenerated overview — check these if "
              "a company has gone missing from a tag query)", flush=True)
    if failed:
        print(f"  Extraction failed for {len(failed)}: {', '.join(failed)}",
              flush=True)
        print("  These keep their previous tags and are retried next run.",
              flush=True)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="re-tag every ticker, ignoring tags_filing_date")
    args = parser.parse_args()
    update_industry_tags(regenerate_all=args.all)
