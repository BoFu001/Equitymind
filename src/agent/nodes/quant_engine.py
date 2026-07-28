"""
src/agent/quant_engine.py

Quant Engine Node — Layer 2 Quantitative Intelligence.

Sits between fetch_all_data (data acquisition) and generate_report (language generation).
Reads ALL inputs from state (stock_snapshots, risk_inputs, quality_inputs,
consensus_inputs, news) — performs NO data fetching of its own. This node
is pure computation only, consistent with the data-layer/compute-layer
separation: fetch_all_data is solely responsible for I/O, quant_engine is
solely responsible for turning that data into signals.

No LLM calls, no network calls — pure Python computation. Fast, deterministic, fully testable.

Signal engines included (grows with each Layer 2 step):
    Step 1: valuation_signal  — P/E + P/B (vs peer group)
    Step 2: momentum_signal   — 12-1 month momentum + 52-week high position
    Step 3: risk_signal       — Beta, Sharpe, VaR, Max Drawdown
    ...
"""

from colors import gprint, mprint
from src.agent.state import AgentState
from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS

from src.quant.valuation_signal import valuation_signal
from src.quant.momentum_signal import momentum_signal

from src.quant.risk_signal import risk_signal
from src.quant.quality_signal import quality_signal
from src.quant.consensus_signal import consensus_signal

from src.quant.news_sentiment_signal import news_sentiment_signal




def quant_engine(state: AgentState) -> dict:
    """
    Compute quantitative signals for all tickers in state["stock_snapshots"].

    For each ticker, runs all available signal engines and aggregates
    results into a structured dict keyed by ticker symbol.

    Args:
        state: AgentState — expects state["stock_snapshots"] to be populated
               by fetch_all_data before this node runs.

    Returns:
        {"quant_signals": {ticker: {signal_name: result, ...}, ...}}
        Empty dict if no market data available.
    """

    stock_snapshots = state.get("stock_snapshots") or {}

    if not stock_snapshots:
        gprint("  [quant_engine] No market data available — skipping")
        return {"quant_signals": {}}

    writer = get_stream_writer()
    writer({"type": "progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_engine"]})

    quant_signals = {}

    all_news              = state.get("news") or {}
    all_risk_inputs       = state.get("risk_inputs") or {}
    all_quality_inputs    = state.get("quality_inputs") or {}
    all_consensus_inputs  = state.get("consensus_inputs") or {}

    for ticker, data in stock_snapshots.items():
        gprint(f"  [quant_engine] Computing signals for {ticker}")

        signals = {}

        # ── Step 1: Valuation Signal ──────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_valuation"].format(ticker=ticker)})
        val = valuation_signal(data)
        if val is not None:
            signals["valuation"] = val
            mprint(
                f"  [valuation] {val['valuation_label']} "
                f"(score={val['valuation_score']}, method={val['method']})"
            )
        else:
            signals["valuation"] = None
            mprint(f"  [valuation] insufficient data")

        # ── Step 2: Momentum Signal ───────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_momentum"].format(ticker=ticker)})
        mom = momentum_signal(ticker)
        if mom is not None:
            signals["momentum"] = mom
            mprint(
                f"  [momentum] 12-1={mom['momentum_12_1_label']} "
                f"(pctile={mom['momentum_12_1_percentile']}) "
                f"52w_position={mom['position_52w_label']} "
                f"(pctile={mom['position_52w_percentile']})"
            )
        else:
            signals["momentum"] = None
            mprint(f"  [momentum] insufficient data")

        # ── Step 3: Risk Signal ───────────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_risk"].format(ticker=ticker)})
        risk_inputs = all_risk_inputs.get(ticker)
        risk = risk_signal(data, risk_inputs)
        if risk is not None:
            signals["risk"] = risk
            mprint(
                f"  [risk] beta_score={risk['beta']['beta_score'] if risk['beta'] else 'N/A'} "
                f"sharpe_score={risk['sharpe']['sharpe_score']} "
                f"var_score={risk['var']['var_score']} "
                f"drawdown_score={risk['max_drawdown']['drawdown_score']}"
            )
        else:
            signals["risk"] = None
            mprint(f"  [risk] insufficient data")

        # ── Step 4: Quality Signal ────────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_quality"].format(ticker=ticker)})
        quality_inputs = all_quality_inputs.get(ticker)
        quality = quality_signal(quality_inputs)
        if quality is not None:
            signals["quality"] = quality
            mprint(
                f"  [quality] {quality['quality_label']} "
                f"(F-Score={quality['f_score_raw']}/{quality['signals_evaluated']}, "
                f"score={quality['quality_score']})"
            )
        else:
            signals["quality"] = None
            mprint(f"  [quality] insufficient data")

        # ── Step 5: News Sentiment Signal ──────────────────────────────────
        # (distinct from a future Management Risk Sentiment Signal, which
        # will measure sentiment in 10-K risk factor sections — hence the
        # explicit "news_" prefix, not a generic "sentiment" key)
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_news_sentiment"].format(ticker=ticker)})
        # News is always pre-fetched and pre-filtered by fetch_all_data
        # (see src/tools/news_reader.py) — this function only scores and
        # aggregates, it no longer needs company_name for filtering.
        news_articles = all_news.get(ticker) or []
        news_sent = news_sentiment_signal(ticker, news_articles)
        if news_sent is not None:
            signals["news_sentiment"] = news_sent
            mprint(
                f"  [news_sentiment] {news_sent['sentiment_label']} "
                f"(score={news_sent['sentiment_score']}, "
                f"n={news_sent['total_articles']})"
            )
        else:
            signals["news_sentiment"] = None
            mprint(f"  [news_sentiment] insufficient data")

        # ── Step 6: Consensus Signal ──────────────────────────────────────
        writer({"type": "sub_progress", "node": "quant_engine", "message": NODE_PROGRESS["quant_consensus"].format(ticker=ticker)})
        consensus_inputs = all_consensus_inputs.get(ticker)
        consensus = consensus_signal(consensus_inputs)
        if consensus is not None:
            signals["consensus"] = consensus
            mprint(
                f"  [consensus] recommendation={consensus['recommendation_label']} "
                f"upside={consensus['upside_label']} "
                f"trend={consensus['trend_label']}"
            )
        else:
            signals["consensus"] = None
            mprint(f"  [consensus] insufficient data")

        quant_signals[ticker] = signals

    gprint(f"  [quant_engine] Signals computed for {list(quant_signals.keys())}")
    return {"quant_signals": quant_signals}