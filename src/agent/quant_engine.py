"""
src/agent/quant_engine.py

Quant Engine Node — Layer 2 Quantitative Intelligence.

Sits between research_loop (data acquisition) and generate_report (language generation).
Reads structured market data from state, computes quantitative signals for each ticker,
and writes results back to state["quant_signals"].

No LLM calls — pure Python computation. Fast, deterministic, fully testable.

Signal engines included (grows with each Layer 2 step):
    Step 1: valuation_signal  — P/E + P/B (vs peer group)
    Step 2: momentum_signal   — coming soon
    Step 3: risk_signal       — Beta, Sharpe, VaR, Max Drawdown
    ...
"""

from colors import gprint
from src.agent.state import AgentState
from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS

from src.quant.valuation_signal import valuation_signal
from src.quant.momentum_signal import momentum_signal

from src.quant.risk_signal import risk_signal
from src.tools.market_data import get_risk_inputs

from src.quant.quality_signal import quality_signal
from src.tools.market_data import get_quality_inputs

from src.quant.consensus_signal import consensus_signal
from src.tools.market_data import get_consensus_inputs




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

    market_data = state.get("market_data") or {}

    if not market_data:
        gprint("  [quant_engine] No market data available — skipping")
        return {"quant_signals": {}}

    writer = get_stream_writer()
    writer({"type": "progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_engine"]})

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
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_momentum"].format(ticker=ticker)})
        mom = momentum_signal(data)
        if mom is not None:
            signals["momentum"] = mom
            gprint(
                f"    momentum: {mom['momentum_label']} "
                f"(score={mom['momentum_score']})"
            )
        else:
            signals["momentum"] = None
            gprint(f"    momentum: insufficient data")

        # ── Step 3: Risk Signal ───────────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_risk"].format(ticker=ticker)})
        risk_inputs = get_risk_inputs(ticker)
        risk = risk_signal(data, risk_inputs)
        if risk is not None:
            signals["risk"] = risk
            gprint(
                f"    risk: beta_score={risk['beta']['beta_score'] if risk['beta'] else 'N/A'} "
                f"sharpe_score={risk['sharpe']['sharpe_score']} "
                f"var_score={risk['var']['var_score']} "
                f"drawdown_score={risk['max_drawdown']['drawdown_score']}"
            )
        else:
            signals["risk"] = None
            gprint(f"    risk: insufficient data")

        # ── Step 4: Quality Signal ────────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_quality"].format(ticker=ticker)})
        quality_inputs = get_quality_inputs(ticker)
        quality = quality_signal(quality_inputs)
        if quality is not None:
            signals["quality"] = quality
            gprint(
                f"    quality: {quality['quality_label']} "
                f"(F-Score={quality['f_score_raw']}/{quality['signals_evaluated']}, "
                f"score={quality['quality_score']})"
            )
        else:
            signals["quality"] = None
            gprint(f"    quality: insufficient data")

        # ── Step 5: Sentiment Signal ──────────────────────────────────────
        # Coming in Step 5
        signals["sentiment"] = None

        # ── Step 6: Consensus Signal ──────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_consensus"].format(ticker=ticker)})
        consensus_inputs = get_consensus_inputs(ticker)
        consensus = consensus_signal(data, consensus_inputs)
        if consensus is not None:
            signals["consensus"] = consensus
            gprint(
                f"    consensus: recommendation={consensus['recommendation_label']} "
                f"upside={consensus['upside_label']} "
                f"trend={consensus['trend_label']}"
            )
        else:
            signals["consensus"] = None
            gprint(f"    consensus: insufficient data")

        quant_signals[ticker] = signals

    gprint(f"  [quant_engine] Signals computed for {list(quant_signals.keys())}")
    return {"quant_signals": quant_signals}