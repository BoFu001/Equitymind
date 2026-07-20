"""
src/agent/nodes/generate_report.py

Node: Generate Report
"""

from datetime import datetime

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters import format_stock_snapshot, format_sec_chunks, format_conversation_context, format_quant_signals
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_report(state: AgentState) -> dict:
    """
    Answers the user's question directly using all available data.
    No fixed template — LLM decides format based on what was asked.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "report", "message": NODE_PROGRESS["generate_report"]})

    question            = state.get("contextualized_question") or state["question"]
    tickers             = state.get("tickers") or []
    all_chunks          = state.get("chunks") or {}
    all_stock_snapshots = state.get("stock_snapshots") or {}
    quant_signals       = state.get("quant_signals") or {}
    messages            = state.get("messages") or []

    # ── Format conversation history ──
    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT)

    # ── Format market data ──
    market_context = "".join(format_stock_snapshot(all_stock_snapshots.get(t, {}), t) for t in tickers)

    # ── Format SEC filing chunks ──
    sec_context = "".join(
        format_sec_chunks(all_chunks.get(t, []), t) if all_chunks.get(t) else f"\n{t}: No SEC 10-K filing available.\n"
        for t in tickers
    )

    # ── Format quant signals ──
    quant_context = "".join(format_quant_signals(quant_signals.get(t, {}), t) for t in tickers)

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

Answer the user's question directly and naturally using the data provided below.
Let the question determine the length and format of your response:
- Simple factual question (e.g. "What is the P/E ratio?", "What about the risk?") → answer in 1-3 sentences with the relevant numbers. No headers, no tables.
- Request for full analysis (e.g. "Analyse NVDA", "Give me a full report on AAPL") → generate a comprehensive structured report with headers, tables, and sections.
- Comparison request (e.g. "Compare NVDA and AMD") → generate a structured side-by-side comparison.
- Discovery request (e.g. "Find me a low risk stock") → rank and recommend from the candidates with real data.

Always use specific numbers from the data. Never be vague.
Always include ALL tickers in the response — never drop any company from the analysis.
Format large numbers cleanly: $24.5B not $24,452,999,168. Round to 2 decimal places.
Use markdown and emojis where appropriate for the format chosen.

GENERAL PRINCIPLES — apply these to EVERY quantitative signal (Valuation,
Momentum, Risk, Quality, and any future signal), not just the specific
cases listed below:
1. Every signal is a historical/backward-looking statistic, never a
   guarantee of future performance. Never phrase a score as a promise
   (e.g. not "this stock cannot fall more than X%" — instead
   "historically, losses have not exceeded X% under normal conditions").
2. If a signal's data is partially missing or a caveat flag is present
   (low_confidence, reference_only, stale_benchmark, signals_evaluated
   < 9, stress_tested=False, Beta=None, etc.), explicitly disclose what
   is missing or uncertain — never silently present a partial or
   qualified signal as if it were complete and unconditional.
3. Present each signal and sub-signal on its own terms — do not label
   combinations of signals with interpretive judgments (e.g. do not call
   any combination a "trap" or a "conflict" requiring resolution). State
   the facts from each signal clearly and let the user draw their own
   conclusions from the full picture. The one exception is a purely
   factual/methodological note: if different valuation methods were used
   for different companies in a comparison (e.g. one uses P/E, another
   uses P/S because it is loss-making), state this explicitly — this is
   a disclosure about measurement comparability, not an interpretive
   judgment about what the numbers mean.
4. Never invent a confident causal explanation (e.g. "why is this
   ratio so high") unless it is directly supported by data in this
   prompt (a specific figure, or SEC filing text). Generic narrative
   ("reflects strong brand value") is speculation, not fact — label it
   as such. Never cite one signal as if it explains another unless the
   data actually shows a link (e.g. Momentum does not explain a
   valuation ratio).
Always state the observation window for Risk metrics (e.g. "based on
the past 2 years of price history") and, when presenting Valuation and
Quality together, remember they are ALWAYS coupled regardless of
question scope — never state a valuation judgment without the
accompanying Quality/F-Score context, even in a short, targeted answer.
5. Consensus Signal (analyst recommendation, upside, trend) reflects
   HUMAN JUDGMENT, not an objective market calculation like the other
   signals — always make this distinction clear. Analyst ratings carry
   a well-documented systematic optimism bias ("sell" ratings are rare
   in practice) — a bullish reading should be described as "positive
   within a system that skews positive," not as a neutral, unbiased
   signal. IMPORTANT: recommendation, upside, and trend are three
   INDEPENDENT sub-signals answering different questions (current
   standing / future price target / recent directional change) — present
   each on its own terms using its own label (e.g. "recommendation is
   bullish, but the trend has been deteriorating"), never average them
   into one overall consensus verdict. The trend_label reflects the
   analyst group's aggregate rating distribution over time, not
   individual analyst revision tracking.

SIGNAL-SPECIFIC FORMATTING NOTES:
- Revenue: MARKET DATA's "revenue" is trailing-twelve-months (TTM), a
  rolling 12-month total — different from any fiscal-year revenue figure
  in SEC filing excerpts. If both appear, label each explicitly (e.g.
  "TTM Revenue: $318.27B" vs "FY2025 Revenue: $281.72B") — presenting
  them unlabeled reads as a contradiction, not two distinct correct figures.
- Analyst sentiment: use recommendation_mean — 1.0–1.5 Strong Buy,
  1.5–2.5 Buy, 2.5–3.5 Hold, 3.5–4.5 Sell, 4.5–5.0 Strong Sell. Do not
  call a mean above 2.5 "bullish". Always mention analyst_count. If the
  target price range is very wide (high > 3x low), flag the divergence.
- Max Drawdown: always state BOTH the peak/trough dates AND prices
  (e.g. "from $257.38 on Dec 26, 2024 to $171.51 on Apr 8, 2025"), even
  if the user only asked about timing.
- F-Score is always out of 9 possible points. If signals_evaluated < 9,
  NEVER write "X/Y" using signals_evaluated as the denominator (e.g.
  "3/7" wrongly implies a 7-point scale) — state both numbers separately:
  "F-Score of 3 out of 9 possible points (only 7 of the 9 signals could
  be evaluated)". If current_ratio_improving or gross_margin_improving
  specifically show as unavailable, this is a known limitation for
  financial institutions (banks, insurers don't report "current assets"
  or "gross profit" the way other companies do) — say so explicitly and
  suggest sector-specific metrics (capital adequacy ratio, net interest
  margin) as a supplement.

USER QUESTION: {question}
TICKERS: {', '.join(tickers)}
DATE: {datetime.now().strftime('%B %d, %Y')}

CONVERSATION HISTORY:
{conversation_context}

MARKET DATA:
{market_context}

SEC FILING DATA:
{sec_context}

QUANTITATIVE SIGNALS:
{quant_context if quant_context else "No quantitative signals available."}"""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    gprint(f"  [generate_report] Response generated for {tickers} ({len(answer)} chars)")
    return {"answer": answer}
