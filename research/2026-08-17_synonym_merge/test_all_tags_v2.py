"""
research/2026-08-17_synonym_merge/test_all_tags_v2.py

Full round-trip test on ALL 257 tags in merged_tag_companies_v2.json
(not a spot check on the top 32 like the earlier test_top_tags.py).
For each tag, wraps it in a minimal query ("{tag} companies") and
checks whether select_tag() resolves back to the same tag with the
same company count.

This uses the v2 vocabulary (post energy/oil-and-gas review, post
health-care/healthcare and other 2026-08-17 merges) — a fresh
select_tag implementation reading from merged_tag_companies_v2.json,
not the older test_tag_selection.py which still points at the
2026-08-16 vocabulary.
"""

import json
from pathlib import Path
from openai import OpenAI
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = Path(__file__).parent / "merged_tag_companies_v2.json"
with open(DATA_PATH) as f:
    tag_data = json.load(f)

TAG_TO_TICKERS = {item["tag"]: item["tickers"] for item in tag_data}
ALL_TAGS = list(TAG_TO_TICKERS.keys())


def select_tag(question: str) -> str | None:
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
    if answer == "NONE" or answer not in TAG_TO_TICKERS:
        return answer
    return answer


if __name__ == "__main__":
    results = []
    print(f"Testing all {len(ALL_TAGS)} tags...\n")

    for i, original_tag in enumerate(ALL_TAGS, 1):
        query = f"{original_tag} companies"
        expected_count = len(TAG_TO_TICKERS[original_tag])
        selected_tag = select_tag(query)

        if selected_tag == "NONE":
            status = "NONE"
            actual_count = None
        elif selected_tag == original_tag:
            status = "OK"
            actual_count = expected_count
        elif selected_tag not in TAG_TO_TICKERS:
            status = "INVALID"
            actual_count = None
        else:
            status = "DIFFERENT"
            actual_count = len(TAG_TO_TICKERS[selected_tag])

        results.append({
            "original_tag": original_tag,
            "expected_count": expected_count,
            "selected_tag": selected_tag,
            "status": status,
        })

        print(f"[{i}/{len(ALL_TAGS)}] {original_tag!r:45s} (expected {expected_count:3d}) -> {selected_tag!r} [{status}]")

    print()
    print("=" * 60)
    ok_count = sum(1 for r in results if r["status"] == "OK")
    different_count = sum(1 for r in results if r["status"] == "DIFFERENT")
    none_count = sum(1 for r in results if r["status"] == "NONE")
    invalid_count = sum(1 for r in results if r["status"] == "INVALID")

    print(f"OK:        {ok_count}/{len(ALL_TAGS)}")
    print(f"DIFFERENT: {different_count}/{len(ALL_TAGS)}")
    print(f"NONE:      {none_count}/{len(ALL_TAGS)}")
    print(f"INVALID:   {invalid_count}/{len(ALL_TAGS)}")

    if different_count > 0:
        print()
        print("DIFFERENT cases:")
        for r in results:
            if r["status"] == "DIFFERENT":
                print(f"  {r['original_tag']!r} -> {r['selected_tag']!r}")

    output_path = Path(__file__).parent / "round_trip_results_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Full results written to: {output_path}")
