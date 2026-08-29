"""
research/2026-08-16_tag_normalization/test_tag_selection.py

Experiment: instead of computing embedding similarity between the
user's industry phrase and each company's overview_embedding, ask the
LLM to pick the single closest tag from the known, deduplicated tag
vocabulary (merged_tag_companies.json), then look up the exact tickers
for that tag directly — no similarity threshold, no candidate pool
truncation needed, because the lookup is exact.

This is a standalone test script, not wired into discovery_experiment.py.
Just prints the LLM's chosen tag and the resulting ticker list for each
test question, so the output can be eyeballed before deciding whether
to build this into the real pipeline.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path so `from config import ...` works when
# this script is run directly from a subfolder (folder name starts
# with a digit, so `python -m` can't be used here).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = Path(__file__).parent / "merged_tag_companies.json"

with open(DATA_PATH) as f:
    tag_data = json.load(f)

TAG_TO_TICKERS = {item["tag"]: item["tickers"] for item in tag_data}
ALL_TAGS = list(TAG_TO_TICKERS.keys())


def select_tag(question: str) -> str | None:
    """
    Asks the LLM to pick the single closest tag from ALL_TAGS for the
    given question. Returns the tag string, or None if the LLM
    declines to match anything.
    """
    tag_list_str = ", ".join(ALL_TAGS)

    prompt = f"""You are matching a user's industry/theme question to the SINGLE closest tag from a fixed vocabulary.

Available tags (pick exactly one, verbatim, from this list):
{tag_list_str}

User's question: "{question}"

Respond with ONLY the exact tag text from the list above that best matches what the user is asking about. If nothing in the list is a reasonable match, respond with exactly: NONE

Your answer (just the tag, nothing else):"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    if answer == "NONE":
        return None
    if answer not in TAG_TO_TICKERS:
        print(f"  [WARNING] LLM returned '{answer}' which is not in the tag vocabulary")
        return None
    return answer


def run_test(question: str):
    print(f"=== \"{question}\" ===")
    tag = select_tag(question)
    if tag is None:
        print("  No matching tag found")
    else:
        tickers = TAG_TO_TICKERS[tag]
        print(f"  Selected tag: '{tag}' ({len(tickers)} companies)")
        print(f"  Tickers: {tickers}")
    print()


if __name__ == "__main__":
    print("### Batch 1: Regular/expected tag matches ###\n")
    run_test("semiconductor stocks")
    run_test("healthcare companies")
    run_test("financial services companies")
    run_test("renewable energy stocks")
    run_test("aerospace and defense companies")

    print("### Batch 2: Requires semantic mapping ###\n")
    run_test("chip stocks")
    run_test("IT companies")
    run_test("spaceship companies")
    run_test("chromatography equipment")
