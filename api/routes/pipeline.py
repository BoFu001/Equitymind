"""
api/routes/pipeline.py

Internal pipeline trigger endpoint — used by Kestra (or any trusted
internal scheduler) to kick off EquityMind's batch data pipeline
scripts, and to poll whether a triggered run has finished.

Not part of the public developer-facing API (see api/auth.py for
that, separate system, separate concern).

Design: the orchestrator (Kestra) owns *when* and *in what order* to
trigger scripts — that responsibility belongs entirely to Kestra, not
duplicated here. equitymind-core only ever answers two questions:
"run this script" and "is this run finished yet". Kestra polls the
second question in a loop before triggering the next dependent
script, which is how the Stage 2 -> Stage 3 -> Stage 4 dependency
chain (see the pipeline operating instructions) gets enforced without
equitymind-core needing to know about that chain at all.

Authentication: a single shared secret (PIPELINE_TRIGGER_SECRET),
checked via a header — not the same mechanism as api/auth.py's
future portal-based API keys, which serve external paying
developers, not internal infrastructure.
"""

import asyncio
import subprocess
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Header

from config import PIPELINE_TRIGGER_SECRET

logger = logging.getLogger(__name__)
router = APIRouter()
# Only scripts explicitly listed here can be triggered — never accept
# an arbitrary script name from the request, to avoid becoming an
# arbitrary-code-execution endpoint.
# All seven pipeline scripts run daily as of 2026-08-11, in the
# execution order below (enforced by Kestra, not by this dict's
# ordering — see the module docstring). build_stock_universe,
# update_common_names, update_peer_groups, and update_stock_overviews
# used to run manually every few months; they're now daily because
# market_cap needs to stay current for Discovery's ranking queries,
# and the other three are incremental (only process newly-added
# tickers), so daily runs cost almost nothing beyond a ticker's first
# day in the universe.
ALLOWED_SCRIPTS = {
    "build_stock_universe": "scripts/build_stock_universe.py",
    "update_common_names": "scripts/update_common_names.py",
    "update_peer_groups": "scripts/update_peer_groups.py",
    "update_stock_overviews": "scripts/update_stock_overviews.py",
    "update_momentum_benchmarks": "scripts/update_momentum_benchmarks.py",
    "update_financial_history": "scripts/update_financial_history.py",
    "update_quant_signals": "scripts/update_quant_signals.py",
}

# In-memory run tracker: {run_id: {status, script, started_at, finished_at, returncode}}.
# Fine for a single-instance deployment — this table is not meant to
# survive a restart, and Kestra polls it while equitymind-core is
# continuously running, not across deploys.
_RUNS: dict[str, dict] = {}


def _run_script_in_background(run_id: str, script_path: str) -> None:
    """
    Runs a script as a subprocess, blocking only within this
    background thread (not the request handler), then records the
    result in _RUNS once it finishes. Some of these scripts take
    several minutes (e.g. update_momentum_benchmarks.py processes
    ~250 tickers) — the HTTP response has already returned by the
    time this finishes.

    The script's own stdout is captured and re-emitted line by line
    with a [PIPELINE] prefix, rather than letting it print directly —
    this lets a Railway log filter on "[PIPELINE]" surface the full
    run (including per-ticker progress lines like "processed 50/250"),
    not just the start/finish summary. Without this, those progress
    lines print unprefixed and get lost among concurrent user-query
    logs, which was the whole point of adding the prefix in the first
    place.
    """
    process = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        logger.info("[PIPELINE] %s", line.rstrip())

    process.wait()

    _RUNS[run_id]["status"] = "success" if process.returncode == 0 else "failed"
    _RUNS[run_id]["returncode"] = process.returncode
    _RUNS[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("[PIPELINE] Run %s (%s) finished with status: %s", run_id, script_path, _RUNS[run_id]["status"])


@router.post("/internal/run-pipeline/{script_name}")
async def run_pipeline_script(script_name: str, x_pipeline_secret: str = Header(None)):
    """
    Triggers one of the allowed batch pipeline scripts, in the
    background. Returns immediately with a run_id — does not wait
    for the script to finish. Call GET /internal/run-pipeline/{run_id}
    to check whether it has completed.
    """
    if x_pipeline_secret != PIPELINE_TRIGGER_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing pipeline secret")

    if script_name not in ALLOWED_SCRIPTS:
        raise HTTPException(status_code=404, detail=f"Unknown script: {script_name}")

    script_path = ALLOWED_SCRIPTS[script_name]
    run_id = str(uuid.uuid4())

    _RUNS[run_id] = {
        "status": "running",
        "script": script_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "returncode": None,
    }

    logger.info("[PIPELINE] Triggering: %s (run_id=%s)", script_path, run_id)

    asyncio.get_event_loop().run_in_executor(None, _run_script_in_background, run_id, script_path)

    return {"status": "started", "script": script_name, "run_id": run_id}


@router.get("/internal/run-pipeline/{run_id}")
async def get_pipeline_run_status(run_id: str, x_pipeline_secret: str = Header(None)):
    """
    Returns the current status of a previously triggered run —
    "running", "success", or "failed". Kestra polls this before
    triggering the next script in a dependency chain.
    """
    if x_pipeline_secret != PIPELINE_TRIGGER_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing pipeline secret")

    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    return _RUNS[run_id]
