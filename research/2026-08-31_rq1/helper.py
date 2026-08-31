"""
Helper for the two hand-judged rules.  It does not give final verdicts; it
lays out what to look at and marks the lines that need your eyes.

  python3 research/2026-08-31_rq1/helper.py

Rule 4 (number on the wrong company): every report with 2+ companies.
   A number belongs to the nearest company name to its LEFT in the line.
Rule 1 (invented number): 15 reports drawn with seed 42.
"""
import glob
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}|"                                   # 2026-08-31
                  r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}")  # Aug 31, 2026


def load_reports():
    reports = []
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.json"))):
        r = json.load(open(path))
        if r["report_prompt"]:
            reports.append(r)
    return reports


def company_blocks(prompt, tickers):
    """The text of each company's own blocks in the prompt (snapshot + signals)."""
    blocks = {}
    for t in tickers:
        parts = re.findall(rf"^{t} — .*?(?=^\w+ — |^SELECTION SCOPE|^QUANTITATIVE SIGNALS|\Z)",
                           prompt, re.S | re.M)
        parts += re.findall(rf"^{t} Quantitative Signals:.*?(?=^\w+ Quantitative Signals:|\Z)",
                            prompt, re.S | re.M)
        blocks[t] = "\n".join(parts)
    return blocks


def in_prompt(number, text):
    """Numeric match. Accepts the same value written differently:
    rounded (28.577 -> 28.58), as a percent (0.40305 -> 40.31),
    or scaled to billions / trillions (4665759498240 -> 4.67)."""
    if number in text or number.replace(",", "") in text:
        return True
    try:
        n = float(number.replace(",", ""))
    except ValueError:
        return False
    decimals = len(number.split(".")[1]) if "." in number else 0
    for raw in NUMBER.findall(text):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        for candidate in (v, v * 100, v / 1e6, v / 1e9, v / 1e12):
            if abs(candidate - n) <= 0.5 * 10 ** (-decimals) + 1e-9:
                return True
    return False


def worth_checking(number):
    """Skip small bare integers (6/9 scores, '12-1', percentiles): they collide everywhere."""
    return "." in number or len(number.lstrip("-")) >= 3


PEERS = re.compile(r"\(peers?\b[^)]*\)", re.I)     # peer lists name other companies


def numbers_with_owner(line, tickers):
    """Each number in the line with the company it belongs to:
    a company named right after the number ('-2.72% AAPL') wins,
    otherwise the nearest company name to the left."""
    clean = DATE.sub(" ", line)
    clean = PEERS.sub(lambda m: " " * len(m.group(0)), clean)   # blank out, keep positions
    mentions = sorted((m.start(), t) for t in tickers for m in re.finditer(rf"\b{t}\b", clean))
    out = []
    for m in NUMBER.finditer(clean):
        # a company name directly after the number ("-2.72% AAPL") wins,
        # but only if nothing except %, *, x or spaces sits in between
        right = [t for pos, t in mentions
                 if m.end() <= pos <= m.end() + 8
                 and set(clean[m.end():pos]) <= set("%*x ")]
        left = [t for pos, t in mentions if pos < m.start()]
        owner = right[0] if right else (left[-1] if left else None)
        out.append((m.group(0), owner))
    return out


def rule4_layout(r):
    print(f"\n=== RULE 4  {r['id']} run {r['run']}  companies {r['tickers']} ===")
    blocks = company_blocks(r["report_prompt"], r["tickers"])
    look = undecidable = unmatched = 0
    for line in r["answer"].splitlines():
        pairs = numbers_with_owner(line, r["tickers"])
        if not pairs or all(owner is None for _, owner in pairs):
            continue
        flagged = []
        for n, owner in pairs:
            if owner is None or not worth_checking(n):
                continue
            found_in = [t for t in r["tickers"] if in_prompt(n, blocks[t])]
            if found_in == [owner]:
                continue                                        # ok, no need to print
            if len(found_in) > 1:
                undecidable += 1
                flagged.append(f"{n} -> {owner}: UNDECIDABLE, number is in blocks of {found_in}")
            elif not found_in:
                unmatched += 1
                flagged.append(f"{n} -> {owner}: not in any block (rounded? derived? check)")
            else:
                look += 1
                flagged.append(f"{n} -> {owner}: LOOK, found only in {found_in[0]}'s block")
        if flagged:
            print(f"\n  {line.strip()[:120]}")
            for f in flagged:
                print(f"      {f}")
    if look == 0 and unmatched == 0:
        print("  every decidable number sits under its own company -> pass")
    return look, undecidable, unmatched


def rule1_layout(r):
    print(f"\n=== RULE 1  {r['id']} run {r['run']} ===")
    clean = DATE.sub(" ", r["answer"])
    missing = [n for n in NUMBER.findall(clean)
               if worth_checking(n) and not in_prompt(n, r["report_prompt"])]
    if not missing:
        print("      every number found in the prompt -> pass")
    for n in missing:
        print(f"      {n:>12}  NOT FOUND -> check by eye: invented / derived / rewritten")
    return missing


def main():
    reports = load_reports()

    multi = [r for r in reports if len(r["tickers"]) >= 2]
    print(f"RULE 4: {len(multi)} reports with 2+ companies")
    rule4 = {}
    for r in multi:
        look, undecidable, unmatched = rule4_layout(r)
        rule4[f"{r['id']}_{r['run']}"] = (
            {"verdict": "pass", "undecidable": undecidable,
             "note": "every decidable number found in its own company's block"}
            if look == 0 and unmatched == 0 else
            {"verdict": "check", "undecidable": undecidable,
             "note": f"{look} LOOK and {unmatched} unmatched number(s) above -- decide by eye"})

    sample = random.Random(42).sample(reports, 15)
    sample.sort(key=lambda r: (r["id"], r["run"]))
    print(f"\n\nRULE 1: 15 reports drawn with seed 42: {[r['id'] + '_' + str(r['run']) for r in sample]}")
    rule1 = {}
    for r in sample:
        missing = rule1_layout(r)
        rule1[f"{r['id']}_{r['run']}"] = (
            {"verdict": "pass", "invented": 0, "derived": 0, "rewritten": 0, "note": "every number found"}
            if not missing
            else {"verdict": "check", "invented": 0, "derived": 0, "rewritten": 0,
                  "note": f"not matched: {missing} -- classify each by eye, then set verdict"})

    template = {"rule4": rule4, "rule1": rule1}
    out = os.path.join(HERE, "manual_verdicts.json")
    if not os.path.exists(out):
        with open(out, "w") as f:
            json.dump(template, f, indent=2)
        print(f"\nTemplate written: {out}  (fill it in by hand)")
    else:
        print(f"\n{out} already exists, not overwritten")


if __name__ == "__main__":
    main()
