"""
Draw the 30 questions for RQ1.  Rules are in questions.md.
Run once, from this folder:  python3 sample_questions.py
Output: questions.json
"""
import json
import os
import random
import re

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script is in
SOURCE = os.path.join(HERE, "..", "user_research") + "/"

# The five companies in the named-stock questions.  Used so we do not
# pick two questions about the same company inside one pattern.
COMPANY_WORDS = {
    "apple": "AAPL", "aapl": "AAPL",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "amazon": "AMZN", "amzn": "AMZN",
    "microsoft": "MSFT", "msft": "MSFT",
}

RANKING_WORDS = ["best", "top", "cheapest", "undervalued", "overvalued",
                 "which", "what are"]

STRESS_QUESTIONS = [
    "What is the strongest HVAC stock?",
    "Which aviation company has the best momentum?",
    "Which healthcare companies have the best valuation?",
    "Which semiconductor stocks are the most overvalued?",
    "Compare the valuation of AAPL, MSFT and GOOGL.",
    "Which is more expensive right now, NVDA, AMD or AVGO?",
]


def read_pool_a():
    """Return {pattern_number: [question, ...]} from questions_by_category.txt."""
    text = open(SOURCE + "questions_by_category.txt").read()
    text = text.split("\n=====", 2)[2]      # drop the header block
    text = text.split("\n=====", 1)[0]      # drop the summary block at the end
    parts = re.split(r"PATTERN (\d+) — ", text)[1:]   # [number, body, number, body, ...]
    pools = {}
    for number, body in zip(parts[0::2], parts[1::2]):
        questions = body.split("Questions:\n", 1)[1]
        lines = [line.strip() for line in questions.splitlines() if line.startswith("  ")]
        pools[int(number)] = lines
    return pools


def company_in(question):
    """Return the ticker of the company named in the question, or None."""
    for word, ticker in COMPANY_WORDS.items():
        if re.search(r"\b" + word + r"\b", question.lower()):
            return ticker
    return None


def draw_pool_a(rng):
    """10 questions: two from patterns 1 and 3, one from each other pattern."""
    pools = read_pool_a()
    wanted = {1: 2, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    picked = []
    for pattern, how_many in wanted.items():
        candidates = pools[pattern][:]
        rng.shuffle(candidates)
        used_companies = set()
        got = 0
        for question in candidates:
            company = company_in(question)
            if company in used_companies:
                continue
            used_companies.add(company)
            picked.append({"pool": "A", "pattern": pattern, "text": question})
            got += 1
            if got == how_many:
                break
        assert got == how_many, f"pattern {pattern}: wanted {how_many}, got {got}"
    return picked


def draw_pool_b(rng):
    """14 questions with a ranking word: 5 semiconductor, 5 tech, 4 healthcare."""
    wanted = {"semiconductor": 5, "tech": 5, "healthcare": 4}
    picked = []
    for sector, how_many in wanted.items():
        lines = open(SOURCE + f"quora_{sector}_stocks.txt").read().splitlines()
        candidates = [q.strip() for q in lines
                      if any(word in q.lower() for word in RANKING_WORDS)]
        assert len(candidates) >= how_many, f"{sector}: only {len(candidates)} candidates"
        rng.shuffle(candidates)
        for question in candidates[:how_many]:
            picked.append({"pool": "B", "sector": sector, "text": question})
    return picked


def main():
    rng = random.Random(SEED)
    items = draw_pool_a(rng) + draw_pool_b(rng)
    for item in items:
        item["group"] = "N"
    for question in STRESS_QUESTIONS:
        items.append({"pool": "S", "group": "S", "text": question})
    for i, item in enumerate(items, start=1):
        item["id"] = f"{item['pool']}{i:02d}"
    assert len(items) == 30, len(items)

    with open(os.path.join(HERE, "questions.json"), "w") as f:
        json.dump({"seed": SEED, "questions": items}, f, indent=2)

    for item in items:
        print(item["id"], item["group"], item["text"])


if __name__ == "__main__":
    main()
