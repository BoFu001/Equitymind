"""
Run the RQ1 questions through EquityMind and save every run to a file.
Product code is not changed: the prompt sent to GPT is copied at the
OpenAI client boundary, inside this script only.

  python3 research/2026-08-31_rq1/run_harness.py A02          # one question, one run
  python3 research/2026-08-31_rq1/run_harness.py              # all questions, RUNS_PER_QUESTION each
  python3 research/2026-08-31_rq1/run_harness.py --no-gate    # RQ2 contrast: gate bypassed, one run per
                                                              # question the gate stopped in the normal runs

Output: runs/{id}_{run}.json          (normal)
        runs_nogate/{id}_{run}.json   (--no-gate)
"""
import asyncio
import glob
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)                      # so "from src..." and "config" import

RUNS_PER_QUESTION = 2
REPORT_MARKER = "COMPANY SNAPSHOT:"           # only the report prompt contains this
NO_GATE = "--no-gate" in sys.argv
OUT_DIR = "runs_nogate" if NO_GATE else "runs"

# ---- 1. Copy every prompt sent to GPT (product code untouched) ----
from openai.resources.chat.completions import Completions

captured = []                                  # all GPT calls of the current run
_original_create = Completions.create

def _recording_create(self, *args, **kwargs):
    captured.append(kwargs.get("messages"))
    return _original_create(self, *args, **kwargs)

Completions.create = _recording_create

# ---- 2. Optional: bypass the Discovery gate (RQ2 contrast group) ----
# Product code untouched. The real node runs; if it decided to ask a
# clarifying question, this wrapper lets the question through instead,
# with the industry it parsed and an empty field list -- which is what
# the system did before the gate existed.
if NO_GATE:
    import src.agent.nodes.discovery_preparation as prep
    _original_prep = prep.discovery_preparation

    def _no_gate(state):
        result = _original_prep(state)
        if result.get("clarification_complete") is False:
            query = prep.extract_discovery_query(state["question"])
            return {"clarification_complete": True,
                    "enriched_query": state["question"],
                    "discovery_query": query.model_dump()}
        return result

    prep.discovery_preparation = _no_gate

    # With no ranking field, today's discovery_execution raises
    # (stages[-1] on an empty list) -- the code after the gate assumes
    # the gate ran.  To reproduce the pre-gate path recorded on
    # 2026-08-20 ("Find me tech stocks" returned all 111), the whole
    # pool is handed to the report unranked, with no selection note.
    import src.agent.nodes.discovery_execution as execution
    _original_exec = execution.discovery_execution

    def _no_gate_execution(state):
        query = execution.DiscoveryQuery(**state["discovery_query"])
        if query.fields:
            return _original_exec(state)
        if query.industry is None:
            pool = execution.get_all_tickers()
        else:
            pool = execution.get_industry_tickers(query.industry)
        print(f"  [no-gate] no ranking field: whole pool of {len(pool)} returned unranked")
        return {"tickers": pool, "discovery_note": None}

    execution.discovery_execution = _no_gate_execution

# ---- 3. Load the graph (after the hooks are in place) ----
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
    error = None
    try:
        final = asyncio.run(equitymind_graph.ainvoke(state))
    except Exception as e:                       # e.g. context_length_exceeded with the gate off
        final = {}
        error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  [harness] run failed, recorded as error: {error[:120]}")
    record = {
        "id": question["id"],
        "group": question["group"],
        "run": run_number,
        "question": question["text"],
        "model": LLM_MODEL,
        "time": datetime.now().isoformat(timespec="seconds"),
        "gate": "off" if NO_GATE else "on",
        "top_intent": final.get("top_intent"),
        "sub_intent": final.get("sub_intent"),
        "tickers": final.get("tickers"),
        "discovery_query": final.get("discovery_query"),
        "discovery_note": final.get("discovery_note"),
        "data_scope": final.get("data_scope"),
        "report_prompt": find_report_prompt(),   # None = no report written
        "answer": final.get("answer"),
        "error": error,                          # None unless the graph raised
    }
    path = os.path.join(HERE, OUT_DIR, f"{question['id']}_{run_number}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    status = "ERROR" if error else ("report" if record["report_prompt"] else "no report")
    print(f"{question['id']} run {run_number}: {status} -> {path}")


def gated_question_ids():
    """Questions the gate stopped in the normal runs: DISCOVERY with no report."""
    ids = set()
    for path in glob.glob(os.path.join(HERE, "runs", "*.json")):
        r = json.load(open(path))
        if r["sub_intent"] == "DISCOVERY" and not r["report_prompt"]:
            ids.add(r["id"])
    return ids


def main():
    os.makedirs(os.path.join(HERE, OUT_DIR), exist_ok=True)
    questions = json.load(open(os.path.join(HERE, "questions.json")))["questions"]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if NO_GATE and args:                        # contrast group, one chosen question (for a test)
        question = next(q for q in questions if q["id"] == args[0])
        run_one(question, 1)
    elif NO_GATE:                               # contrast group: every gated question, one run each
        ids = gated_question_ids()
        for question in questions:
            if question["id"] not in ids:
                continue
            if os.path.exists(os.path.join(HERE, OUT_DIR, f"{question['id']}_1.json")):
                print(f"{question['id']}: already done, skipped")
                continue
            run_one(question, 1)
    elif args:                                  # one question, one run
        question = next(q for q in questions if q["id"] == args[0])
        run_one(question, 1)
    else:                                       # everything
        for question in questions:
            for run_number in range(1, RUNS_PER_QUESTION + 1):
                run_one(question, run_number)


if __name__ == "__main__":
    main()
