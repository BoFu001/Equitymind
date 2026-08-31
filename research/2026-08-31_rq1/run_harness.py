"""
Run the RQ1 questions through EquityMind and save every run to a file.
Product code is not changed: the prompt sent to GPT is copied at the
OpenAI client boundary, inside this script only.

  python3 research/2026-09-02_rq1/run_harness.py A02      # one question, one run
  python3 research/2026-09-02_rq1/run_harness.py          # all questions, RUNS_PER_QUESTION each

Output: research/2026-09-02_rq1/runs/{id}_{run}.json
"""
import asyncio
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)                      # so "from src..." and "config" import

RUNS_PER_QUESTION = 2
REPORT_MARKER = "COMPANY SNAPSHOT:"           # only the report prompt contains this

# ---- 1. Copy every prompt sent to GPT (product code untouched) ----
from openai.resources.chat.completions import Completions

captured = []                                  # all GPT calls of the current run
_original_create = Completions.create

def _recording_create(self, *args, **kwargs):
    captured.append(kwargs.get("messages"))
    return _original_create(self, *args, **kwargs)

Completions.create = _recording_create

# ---- 2. Load the graph (after the copy hook is in place) ----
from config import LLM_MODEL
from src.agent.graph import equitymind_graph
from src.agent.state import build_initial_state


def find_report_prompt():
    """Return the prompt of the report call, or None if no report was written."""
    hits = [m[0]["content"] for m in captured if m and REPORT_MARKER in m[0]["content"]]
    assert len(hits) <= 1, f"expected at most one report call, got {len(hits)}"
    return hits[0] if hits else None


def run_one(question, run_number):
    captured.clear()
    state = build_initial_state(question["text"])
    final = asyncio.run(equitymind_graph.ainvoke(state))
    record = {
        "id": question["id"],
        "group": question["group"],
        "run": run_number,
        "question": question["text"],
        "model": LLM_MODEL,
        "time": datetime.now().isoformat(timespec="seconds"),
        "top_intent": final.get("top_intent"),
        "sub_intent": final.get("sub_intent"),
        "tickers": final.get("tickers"),
        "discovery_query": final.get("discovery_query"),
        "discovery_note": final.get("discovery_note"),
        "data_scope": final.get("data_scope"),
        "report_prompt": find_report_prompt(),   # None = no report written
        "answer": final.get("answer"),
    }
    path = os.path.join(HERE, "runs", f"{question['id']}_{run_number}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"{question['id']} run {run_number}: {'report' if record['report_prompt'] else 'no report'} -> {path}")


def main():
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    questions = json.load(open(os.path.join(HERE, "questions.json")))["questions"]

    if len(sys.argv) > 1:                       # one question, one run
        wanted = sys.argv[1]
        question = next(q for q in questions if q["id"] == wanted)
        run_one(question, 1)
    else:                                       # everything
        for question in questions:
            for run_number in range(1, RUNS_PER_QUESTION + 1):
                run_one(question, run_number)


if __name__ == "__main__":
    main()
