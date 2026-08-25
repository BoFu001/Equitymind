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
from src.agent.formatters.snapshot_formatter import format_stock_snapshot
from src.agent.formatters.sec_formatter import format_sec_chunks
from src.agent.formatters.conversation_formatter import format_conversation_context
from src.agent.formatters.valuation_formatter import format_valuation
from src.agent.formatters.momentum_formatter import format_momentum
from src.agent.formatters.risk_formatter import format_risk
from src.agent.formatters.quality_formatter import format_quality
from src.agent.formatters.news_sentiment_formatter import format_news_sentiment
from src.agent.formatters.consensus_formatter import format_consensus
from src.agent.formatters.short_formatter import format_short
from src.agent.formatters.financial_history_formatter import format_financial_history
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)

# Presentation rules keyed by data_scope entry — only the ones in
# data_scope are included in the final prompt, so a narrow question
# (e.g. only Consensus) doesn't carry rules about Valuation P/E, Max
# Drawdown, or F-Score that don't apply to anything actually shown.
# Not every entry is a quant signal: financial_history and sec_filing
# are data sources, which is why this is keyed by scope rather than
# by signal.
DATA_SCOPE_NOTES = {
    "valuation": """\
- Valuation: if the data includes "(peers: X, Y, Z)" (named peer
  companies), you MUST name those specific companies in your answer
  (e.g. "trading at a premium to peers like Microsoft, Alphabet, and
  Meta") — do not compress this into a generic phrase like "peer
  average" or "industry peers" with no names, even in a short answer.
  If instead the data says "S&P 500 average...(no peer-specific data
  available)", state explicitly that no company-specific peer data was
  available and a broad market average was used instead — this is a
  materially different, lower-confidence comparison and must not be
  presented the same way as a named peer match.
""",
    "risk": """\
- Always state the observation window for Risk metrics (e.g. "based
  on the past 2 years of price history").
- Max Drawdown: always state BOTH the peak/trough dates AND prices
  (e.g. "from $257.38 on Dec 26, 2024 to $171.51 on Apr 8, 2025"), even
  if the user only asked about timing.
""",
    "quality": """\
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
""",
    "consensus": """\
- Consensus Signal (analyst recommendation and upside) reflects
  HUMAN JUDGMENT, not an objective market calculation like the other
  signals — always make this distinction clear. Analyst ratings carry
  a well-documented systematic optimism bias ("sell" ratings are rare
  in practice) — a bullish reading should be described as "positive
  within a system that skews positive," not as a neutral, unbiased
  signal. IMPORTANT: recommendation and upside are two INDEPENDENT
  sub-signals answering different questions (where the rating stands
  now / how far the target price sits above the current one) — present
  each on its own terms using its own label (e.g. "analysts rate it a
  buy, though the target price implies only modest upside"), never
  average them into one overall consensus verdict.
- Analyst sentiment: use recommendation_mean — 1.0-1.5 Strong Buy,
  1.5-2.5 Buy, 2.5-3.5 Hold, 3.5-4.5 Sell, 4.5-5.0 Strong Sell. Do not
  call a mean above 2.5 "bullish". Always mention analyst_count. If the
  target price range is very wide (high > 3x low), flag the divergence.
""",
    "short": """\
- Short Interest data is a SLOW-updating signal (approximately monthly,
  from FINRA/NASDAQ official reporting) - ALWAYS state the reporting
  date shown in the data alongside any percentage or ratio; a number
  with no date is not meaningfully interpretable.
- The three components carry DIFFERENT evidentiary weight - never
  present them with equal confidence. Short Interest level has
  well-established peer-reviewed academic support (Asquith, Pathak and
  Ritter 2005; Boehmer, Jones and Zhang 2008) and can be stated as a
  primary signal. Days to Cover is a liquidity/squeeze-risk indicator
  ONLY, not a directional predictor - its "elevated" threshold is an
  industry-practice convention, not a peer-reviewed statistic; never
  phrase it as "more bearish". Month-over-month change is DESCRIPTIVE
  CONTEXT ONLY - academic evidence on whether it predicts returns is
  mixed and inconclusive; never present it as an independent signal
  or combine it with the other two into one verdict.
""",
    "news": """\
- News Sentiment reflects MEDIA TONE (how recent news articles are
  written about the company), not analyst opinion (Consensus) or an
  objective financial calculation like Valuation/Momentum/Risk/Quality
  — always make this distinction clear. A sentiment score is an
  interpretation of language in news coverage, not a fact about the
  company's fundamentals or future performance.
""",
    # No sec_filing note yet — pending a systematic review of what SEC
    # free text can get wrong, rather than one rule per failure found.
    # Known failure, reproduced twice on 2026-08-10: asked for ADP's
    # fiscal 2023 R&D, the model answered $1.276 billion, which the
    # filing attributes to fiscal 2024 in a three-year series ("fiscal
    # years ended June 30, 2026, 2025 and 2024 ... $1.405bn, $1.388bn,
    # $1.276bn"). The filing holds nothing for fiscal 2023 — position in
    # the series is the only thing tying each figure to a year.
    "financial_history": """\
- HISTORICAL FINANCIALS lists every period this system holds for the
  company, and within each period every metric is either a figure or
  "N/A". "N/A" means the number is not available — not zero, not
  approximately something, not a value to supply from your own
  knowledge.
- Judge availability PER FIGURE, never per year. A period appearing in
  the list does not mean every metric for it is present: the data
  source publishes EPS ahead of the full statements, banks report no
  cost of revenue, and companies paying no dividend have no dividend
  line. Equally, a metric being present for one period says nothing
  about the next.
- If the user asks for a figure that reads N/A, or for a period not in
  the list at all, state that the data does not contain it and that
  you cannot verify a value for it. This applies even if you believe
  you know the correct figure — an unverified number from training
  data cannot be checked against a source, so it is not an acceptable
  substitute for data this system actually retrieved. It applies to
  financial statement figures (revenue, EBITDA, net income, etc.) for
  any specific past fiscal year or quarter.
""",
    # Known unresolved behaviour: when the newest period held has a
    # figure of N/A and the question uses a relative reference ("last
    # fiscal year", "most recent quarter"), the model quotes the newest
    # period that does carry a figure without saying the newer one is
    # empty — reading as though nothing newer had been filed. Tried and
    # reverted on 2026-08-10: an explicit rule for relative references,
    # naming the newest period in the formatter header, and
    # temperature=0. None held across repeated runs (roughly one in
    # four either way), so none is in the code. The figure given is
    # correct throughout; only the disclosure is missing.
}


def generate_report(state: AgentState) -> dict:
    """
    Answers the user's question directly using all available data.
    No fixed template — LLM decides format based on what was asked.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "report", "message": NODE_PROGRESS["generate_report"]})

    question              = state.get("contextualized_question") or state["question"]
    tickers               = state.get("tickers") or []
    messages              = state.get("messages") or []
    data_scope            = state["data_scope"]
    all_stock_snapshots   = state.get("stock_snapshots") or {}
    all_chunks            = state.get("chunks") or {}
    quant_signals         = state.get("quant_signals") or {}
    all_financial_history = state.get("financial_history_data") or {}

    # ── Format conversation history ──
    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT)

    # ── Format company snapshot, SEC filing chunks, quant signals, and
    # historical financials — one loop over tickers instead of four,
    # each ticker only processed once per section ──
    snapshot_parts = []
    sec_parts = []
    quant_parts = []
    financial_history_parts = []

    for t in tickers:
        snapshot_parts.append(format_stock_snapshot(all_stock_snapshots.get(t, {}), t))

        # Quant signals: each formatter called directly and
        # independently, only for the signals determine_data_scope
        # decided this question needs — not going through a fixed
        # bundler (format_quant_signals). See quant_signals_formatter.py's
        # docstring (2026-07-27) for why this was always the planned
        # end state, deferred until this routing node existed.
        signals = quant_signals.get(t, {})
        signal_parts = []
        if "valuation" in data_scope:
            signal_parts.append(format_valuation(signals.get("valuation")))
        if "momentum" in data_scope:
            signal_parts.append(format_momentum(signals.get("momentum")))
        if "risk" in data_scope:
            signal_parts.append(format_risk(signals.get("risk")))
        if "quality" in data_scope:
            signal_parts.append(format_quality(signals.get("quality")))

        if "news" in data_scope:
            signal_parts.append(format_news_sentiment(signals.get("news_sentiment")))
        if "consensus" in data_scope:
            signal_parts.append(format_consensus(signals.get("consensus")))
        if "short" in data_scope:
            signal_parts.append(format_short(signals.get("short")))

        if signal_parts:
            ticker_quant_text = f"\n{t} Quantitative Signals:\n" + "".join(signal_parts)
            quant_parts.append(ticker_quant_text)

        # Historical financials: fetched by fetch_data.py's
        # _fetch_financial_history() (moved there from this file on
        # 2026-07-27 — a report-generation node calling a database
        # reader directly violated the data-layer/display-layer
        # separation). Skipped entirely when not needed.
        if "financial_history" in data_scope:
            financial_history_parts.append(format_financial_history(all_financial_history.get(t, []), t))
        # SEC filing chunks: fetched by fetch_data.py only when
        # "sec_filing" is in data_scope. Skipped entirely when not
        # needed, matching the pattern used by every other conditional
        # section above.
        if "sec_filing" in data_scope:
            sec_parts.append(
                format_sec_chunks(all_chunks.get(t, []), t) if all_chunks.get(t)
                else f"\n{t}: No SEC 10-K filing found.\n"
            )

    snapshot_context = "".join(snapshot_parts)
    sec_context = "".join(sec_parts)
    quant_context = "".join(quant_parts)
    financial_history_context = "".join(financial_history_parts)

    # ── Assemble prompt sections that should disappear entirely when
    # empty (not even a header), rather than showing an empty section
    # or a "No X available" placeholder ──
    financial_history_section = (
        "HISTORICAL FINANCIALS (reported figures by fiscal year and\n"
        "quarter — the source for any question naming a financial\n"
        "statement line item, for one period or across several):\n"
        f"{financial_history_context}\n\n"
        if financial_history_context else ""
    )
    quant_section = (
        f"QUANTITATIVE SIGNALS:\n{quant_context}\n\n"
        if quant_context else ""
    )
    sec_section = (
        f"SEC FILING DATA:\n{sec_context}"
        if sec_context else ""
    )

    scope_notes = "".join(
        DATA_SCOPE_NOTES[s] for s in data_scope if s in DATA_SCOPE_NOTES
    )

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

GENERAL PRINCIPLES — apply these to EVERY quantitative signal shown
below, not just the specific cases listed:
1. Every signal is a historical/backward-looking statistic, never a
   guarantee of future performance. Never phrase a score as a promise
   (e.g. not "this stock cannot fall more than X%" — instead
   "historically, losses have not exceeded X% under normal conditions").
2. If a signal's data is partially missing or a caveat flag is present
   (low_confidence, reference_only, signals_evaluated < 9,
   stress_tested=False, Beta=None, etc.), explicitly disclose what
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
5. The data below was selected for this question and is all there is.
   If the user asks about something that is not in it, say it was not
   retrieved — do not answer from the snapshot instead. Price and market
   cap do not establish whether a company is expensive, risky or well
   run; those come from signals, and a signal that is absent is absent.

FORMATTING NOTES FOR THE DATA BELOW (pre-filtered to this question's
scope, so every rule here applies to something actually shown):
{scope_notes}- Revenue: COMPANY SNAPSHOT's "revenue" is trailing-twelve-months (TTM), a
  rolling 12-month total — different from any fiscal-year revenue figure
  in SEC filing excerpts. If both appear, label each explicitly (e.g.
  "TTM Revenue: $318.27B" vs "FY2025 Revenue: $281.72B") — presenting
  them unlabeled reads as a contradiction, not two distinct correct figures.

USER QUESTION: {question}
TICKERS: {', '.join(tickers)}
DATE: {datetime.now().strftime('%B %d, %Y')}

CONVERSATION HISTORY:
{conversation_context}

COMPANY SNAPSHOT:
{snapshot_context}

{financial_history_section}{quant_section}{sec_section}"""

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
