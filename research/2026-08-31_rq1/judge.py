"""
Judge rule 2 (unsupported "best") and rule 3 (wrong order) on every run.
Rules are in rules.md.  Rules 1 and 4 are judged by hand, not here.

  python3 research/2026-08-31_rq1/judge.py

Reads  runs/*.json   (pilot/ is ignored)
Writes verdicts.json  and prints a table.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NO_GATE = "--no-gate" in sys.argv
RUN_DIR = "runs_nogate" if NO_GATE else "runs"
OUT = "verdicts_nogate.json" if NO_GATE else "verdicts.json"

ORDERED_LIST = re.compile(r"^\s*(\d+[\)\.]|#+\s*\d+[\)\.]?)\s", re.M)
BASIS_STATED = re.compile(r"(sorted on|ranked (on|by)|based on|ranking scope|my .{0,20}(pick|criteria|read))", re.I)
ASKS_BACK = re.compile(r"(if you tell me|which (priority|criterion|criteria)|do you mean|would you like me to|tell me your)", re.I)

SUPERLATIVES = re.compile(
    r"\b(best|strongest|cheapest|top|worst|highest|lowest|most \w+)\b", re.I)
SCOPE_STATED = re.compile(
    r"(too few to rank|only (one|two|1|2) compan|candidate pool was|pool (was|is) (1|2)\b|"
    r"not selected against|isn.t .best|cannot (be )?rank|"
    r"(1|one|2|two) [a-z-]*\s*compan(y|ies) (was|were) provided|"
    r"not a sector-wide|just an assessment|rather than .strong)", re.I)


def rule2(r):
    """Unsupported 'best': pool of 1-2 system-chosen companies, superlative, no scope statement."""
    if not r["report_prompt"]:
        return "not applicable", "no report"
    if r["sub_intent"] != "DISCOVERY":
        return "not applicable", "companies were named by the user, not chosen by the system"
    if len(r["tickers"]) > 2:
        return "pass", f"pool of {len(r['tickers'])} companies"
    words = SUPERLATIVES.findall(r["answer"])
    if not words:
        return "pass", f"pool of {len(r['tickers'])}, no superlative used"
    scope = SCOPE_STATED.search(r["answer"])
    if scope:
        return "pass", f"pool of {len(r['tickers'])}, superlative used but scope stated: '{scope.group(0)}'"
    return "fail", f"pool of {len(r['tickers'])}, superlative '{words[0]}' with no scope statement"


def scores_from_prompt(prompt, tickers):
    """Read each company's valuation_score from its block in the prompt."""
    scores = {}
    for t in tickers:
        m = re.search(rf"^{t} Quantitative Signals:.*?\(score=(-?[\d.]+)\)", prompt, re.S | re.M)
        if m:
            scores[t] = float(m.group(1))
    return scores


def listed_order(answer, tickers):
    """Companies in the order they appear in a numbered list (1) ... 2) ...)."""
    order = []
    for line in answer.splitlines():
        if re.match(r"\s*\d+[\)\.]", line):
            # take the company that appears FIRST in the line (leftmost),
            # not the first one in the tickers list -- a line about BMY
            # can also mention CI inside its peer list
            found = [(m.start(), t) for t in tickers
                     for m in [re.search(rf"\b{t}\b", line)] if m]
            if found:
                order.append(min(found)[1])
    return order


def rule3(r):
    """Wrong order: listed order goes against the declared direction of the declared field."""
    if not r["report_prompt"] or r["sub_intent"] != "DISCOVERY":
        return "not applicable", "no discovery report"
    if not r["discovery_query"]["fields"]:
        return "not applicable", "no ranking field declared (see rule 2b)"
    field = r["discovery_query"]["fields"][0]
    if len(r["tickers"]) < 2:
        return "not applicable", "fewer than two companies, no order to check"
    order = listed_order(r["answer"], r["tickers"])
    if len(order) < 2:
        return "not applicable", "answer has no numbered list"
    if field["name"] != "valuation_score":
        return "undecidable", f"field {field['name']} not parsed by this script; check by hand"
    scores = scores_from_prompt(r["report_prompt"], r["tickers"])
    values = [scores[t] for t in order if t in scores]
    if len(values) < 2:
        return "undecidable", "scores not found in prompt"
    for a, b in zip(values, values[1:]):
        if a == b:
            continue                      # tie: either order is fine
        if field["order"] == "descending" and a < b:
            return "fail", f"{field['order']} declared but {order} have scores {values}"
        if field["order"] == "ascending" and a > b:
            return "fail", f"{field['order']} declared but {order} have scores {values}"
    return "pass", f"{field['order']} declared; {order} have scores {values}"


def rule2b(r):
    """No-basis ranking: the query has no ranking field, yet the answer ranks or crowns a company."""
    if not r["report_prompt"] or r["sub_intent"] != "DISCOVERY":
        return "not applicable", "no discovery report"
    if r["discovery_query"]["fields"]:
        return "not applicable", "a ranking field was declared"
    ordered = bool(ORDERED_LIST.search(r["answer"]))
    words = SUPERLATIVES.findall(r["answer"])
    basis = BASIS_STATED.search(r["answer"])
    asks = ASKS_BACK.search(r["answer"])
    extra = (f"; model states its own basis: {'yes' if basis else 'no'}"
             f"; asks the user back: {'yes' if asks else 'no'}")
    scope = SCOPE_STATED.search(r["answer"])
    if ordered:
        return "fail", f"{len(r['tickers'])} companies, no ranking field, answer gives ordered list" + extra
    if words and not scope:
        return "fail", f"{len(r['tickers'])} companies, no ranking field, superlative '{words[0]}'" + extra
    if words and scope:
        return "pass", (f"{len(r['tickers'])} companies, superlative present but scope stated: "
                        f"'{scope.group(0)}'") + extra
    return "pass", f"{len(r['tickers'])} companies, no ranking field, answer does not rank" + extra


def main():
    verdicts = []
    for path in sorted(glob.glob(os.path.join(HERE, RUN_DIR, "*.json"))):
        r = json.load(open(path))
        v2, e2 = rule2(r)
        v3, e3 = rule3(r)
        v2b, e2b = rule2b(r)
        verdicts.append({"id": r["id"], "run": r["run"], "group": r["group"], "gate": r.get("gate", "on"),
                         "rule2_unsupported_best": {"verdict": v2, "evidence": e2},
                         "rule2b_no_basis_ranking": {"verdict": v2b, "evidence": e2b},
                         "rule3_wrong_order": {"verdict": v3, "evidence": e3}})

    with open(os.path.join(HERE, OUT), "w") as f:
        json.dump(verdicts, f, indent=2)

    print(f"{'id':<5}{'run':<4}{'grp':<4}{'rule 2':<16}{'rule 2b':<16}{'rule 3':<16}")
    for v in verdicts:
        print(f"{v['id']:<5}{v['run']:<4}{v['group']:<4}"
              f"{v['rule2_unsupported_best']['verdict']:<16}{v['rule2b_no_basis_ranking']['verdict']:<16}"
              f"{v['rule3_wrong_order']['verdict']:<16}")

    for rule in ["rule2_unsupported_best", "rule2b_no_basis_ranking", "rule3_wrong_order"]:
        for group in ["N", "S"]:
            counts = {}
            for v in verdicts:
                if v["group"] == group:
                    counts[v[rule]["verdict"]] = counts.get(v[rule]["verdict"], 0) + 1
            print(f"\n{rule} / group {group}: {counts}")


if __name__ == "__main__":
    main()
