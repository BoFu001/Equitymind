"""
Build the RQ1 results table from verdicts.json (rules 2, 3) and
manual_verdicts.json (rules 1, 4).  Writes results.md and prints it.

  python3 research/2026-08-31_rq1/table.py
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(fails, n):
    """95% confidence interval for a rate, as a string.  Empty if n is 0."""
    if n == 0:
        return "-"
    z = 1.96
    p = fails / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{max(0, centre - half):.0%} to {min(1, centre + half):.0%}"


def main():
    auto = json.load(open(os.path.join(HERE, "verdicts.json")))
    manual = json.load(open(os.path.join(HERE, "manual_verdicts.json")))
    group_of = {v["id"]: v["group"] for v in auto}

    # one flat list: (rule, id, run, group, verdict, undecidable_count)
    rows = []
    for v in auto:
        rows.append(("Unsupported best", v["id"], v["run"], v["group"], v["rule2_unsupported_best"]["verdict"], 0))
        rows.append(("Wrong order", v["id"], v["run"], v["group"], v["rule3_wrong_order"]["verdict"], 0))
    for key, v in manual["rule4"].items():
        qid, run = key.split("_")
        rows.append(("Number on wrong company", qid, int(run), group_of[qid], v["verdict"], v.get("undecidable", 0)))
    for key, v in manual["rule1"].items():
        qid, run = key.split("_")
        rows.append(("Invented number", qid, int(run), group_of[qid], v["verdict"], 0))

    lines = ["# RQ1 results", "",
             "60 runs: 30 questions x 2 runs. Group N = 24 real user questions, Group S = 6 stress questions.",
             "Rules 2 and 3 judged by script on all 60 runs; rule 4 on all 14 multi-company reports;",
             "rule 1 by hand on a sample of 15 reports (seed 42).", "",
             "| Rule | Group | Judged | Pass | Fail | Undecidable numbers | Not applicable / not sampled | Runs failed | Questions failed | 95% CI (runs) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for rule in ["Invented number", "Unsupported best", "Wrong order", "Number on wrong company"]:
        for group in ["N", "S"]:
            sub = [r for r in rows if r[0] == rule and r[3] == group]
            judged = [r for r in sub if r[4] in ("pass", "fail")]
            fails = [r for r in sub if r[4] == "fail"]
            total_in_group = sum(1 for v in auto if v["group"] == group)
            if rule in ("Number on wrong company", "Invented number"):
                na = ["x"] * (total_in_group - len(sub))        # runs not in the manual file
            else:
                na = [r for r in sub if r[4] == "not applicable"]
            undec_runs = [r for r in sub if r[4] == "undecidable"]
            undec_numbers = sum(r[5] for r in sub)
            q_judged = {r[1] for r in judged}
            q_failed = {r[1] for r in fails}
            lines.append(
                f"| {rule} | {group} | {len(judged)} | {len(judged) - len(fails)} | {len(fails)} | "
                f"{undec_numbers}{' (+' + str(len(undec_runs)) + ' runs)' if undec_runs else ''} | {len(na)} | "
                f"{len(fails)}/{len(judged)} | {len(q_failed)}/{len(q_judged)} | {wilson(len(fails), len(judged))} |")

    lines += ["",
              "## Run-to-run agreement",
              "Questions whose two runs got different verdicts on any rule:"]
    disagree = []
    for rule in ["Unsupported best", "Wrong order", "Number on wrong company", "Invented number"]:
        by_q = {}
        for r in rows:
            if r[0] == rule:
                by_q.setdefault(r[1], set()).add(r[4])
        disagree += [f"{rule}: {q}" for q, vs in by_q.items() if len(vs) > 1]
    lines.append("none" if not disagree else "\n".join("- " + d for d in disagree))

    lines += ["",
              "## Reading the table",
              "- 'Judged' counts runs with a pass or fail verdict. 'Not applicable' runs had no report,",
              "  or the rule did not apply (companies named by the user; a single-company report).",
              "  For 'Invented number' the column is runs not in the 15-run sample.",
              "- 'Undecidable numbers' are numbers that appear in more than one company's data block; they are",
              "  counted, not judged. They never add to fails.",
              "- Group S rates are provoked, not real-use rates."]

    text = "\n".join(lines)
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
