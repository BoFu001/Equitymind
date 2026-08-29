"""
research/2026-08-27_model_comparison/field_selection_baseline.py

Does extract_discovery_query pick the right field and direction?

Seven questions with one correct answer each, run five times per model.
Four of them ask for a _score field's unfavourable end — the direction
that cannot be read off the wording and has to come from knowing how
the field is signed. On 2026-08-26 "which stock is the most expensive?"
returned market_cap (descending) under gpt-4o: the ranking then covered
the largest companies, not the priciest, and the report answered from
P/E anyway so the substitution never showed.

The other three are controls: a raw percentage read literally, the one
field whose scale runs the other way, and market_cap on a question that
genuinely does mean size.

Run before and after changing LLM_MODEL. Results in results/.
"""
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import LLM_MODEL
from src.agent.nodes.discovery_preparation import extract_discovery_query

CASES = [
    ("which stock is the most expensive?",         "valuation_score",               "ascending"),
    ("which stock is the cheapest?",               "valuation_score",               "descending"),
    ("which stocks are the riskiest?",             "risk_beta_score",               "ascending"),
    ("which stocks have the worst quality?",       "quality_score",                 "ascending"),
    ("which stocks are most heavily shorted?",     "short_interest_pct",            "descending"),
    ("which stocks have the best analyst rating?", "consensus_recommendation_mean", "ascending"),
    ("which stocks are the largest?",              "market_cap",                    "descending"),
]

RUNS = 5


def main() -> None:
    lines = [
        f"model:  {LLM_MODEL}",
        f"run at: {datetime.now():%Y-%m-%d %H:%M}",
        f"runs:   {RUNS} per case",
        "",
    ]

    total_ok = 0
    for question, want_name, want_order in CASES:
        seen = Counter()
        for _ in range(RUNS):
            try:
                query = extract_discovery_query(question)
                field = query.fields[0] if query.fields else None
                got = f"{field.name}/{field.order}" if field else "no field"
            except Exception as exc:
                got = f"error: {type(exc).__name__}"
            seen[got] += 1

        want = f"{want_name}/{want_order}"
        ok = seen[want]
        total_ok += ok

        mark = " " if ok == RUNS else "!"
        lines.append(f"{mark} {question:44} {ok}/{RUNS}  want {want}")
        for got, n in seen.most_common():
            if got != want:
                lines.append(f"    instead: {got} x{n}")

    lines += ["", f"  {total_ok}/{len(CASES) * RUNS}"]

    report = "\n".join(lines)
    print("\n" + report + "\n")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{LLM_MODEL.replace('.', '_')}.txt"
    out_file.write_text(report + "\n")
    print(f"  written to {out_file.relative_to(Path.cwd())}\n")


if __name__ == "__main__":
    main()
