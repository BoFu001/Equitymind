"""
src/agent/quant_engine.py

Quant Engine Node — Layer 2 Quantitative Intelligence.

Sits between research_loop (data acquisition) and generate_report (language generation).
Reads structured market data from state, computes quantitative signals for each ticker,
and writes results back to state["quant_signals"].

No LLM calls — pure Python computation. Fast, deterministic, fully testable.

Signal engines included (grows with each Layer 2 step):
    Step 1: valuation_signal  — P/E + PEG + analyst upside
    Step 2: momentum_signal   — coming soon
    Step 3: risk_signal       — coming soon
    ...
"""

from colors import gprint
from src.agent.state import AgentState
from src.quant.valuation_signal import valuation_signal
from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS




def quant_engine(state: AgentState) -> dict:
    """
    Compute quantitative signals for all tickers in state["market_data"].

    For each ticker, runs all available signal engines and aggregates
    results into a structured dict keyed by ticker symbol.

    Args:
        state: AgentState — expects state["market_data"] to be populated
               by research_loop before this node runs.

    Returns:
        {"quant_signals": {ticker: {signal_name: result, ...}, ...}}
        Empty dict if no market data available.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_engine"]})

    market_data = state.get("market_data") or {}

    if not market_data:
        gprint("  [quant_engine] No market data available — skipping")
        return {"quant_signals": {}}

    quant_signals = {}

    for ticker, data in market_data.items():
        gprint(f"  [quant_engine] Computing signals for {ticker}")

        signals = {}

        # ── Step 1: Valuation Signal ──────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_valuation"].format(ticker=ticker)})
        val = valuation_signal(data)
        if val is not None:
            signals["valuation"] = val
            gprint(
                f"    valuation: {val['valuation_label']} "
                f"(score={val['valuation_score']}, method={val['method']})"
            )
        else:
            signals["valuation"] = None
            gprint(f"    valuation: insufficient data")

        # ── Step 2: Momentum Signal ───────────────────────────────────────
        # Coming in Step 2
        signals["momentum"] = None

        # ── Step 3: Risk Signal ───────────────────────────────────────────
        # Coming in Step 3
        signals["risk"] = None

        # ── Step 4: Quality Signal ────────────────────────────────────────
        # Coming in Step 4
        signals["quality"] = None

        # ── Step 5: Sentiment Signal ──────────────────────────────────────
        # Coming in Step 5
        signals["sentiment"] = None

        # ── Step 6: Consensus Signal ──────────────────────────────────────
        # Coming in Step 6
        signals["consensus"] = None

        quant_signals[ticker] = signals

    gprint(f"  [quant_engine] Signals computed for {list(quant_signals.keys())}")
    return {"quant_signals": quant_signals}