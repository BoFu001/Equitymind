"""
src/agent/formatters/__init__.py

Shared formatters for converting structured data into LLM-readable strings.
Used by fetch_all_data.py and nodes.py to ensure consistent data presentation.

Was a single file (formatters.py) until 2026-07-27, when format_consensus
was split into its own file (consensus.py) — see that file's docstring
for why. Re-exported here so every existing `from src.agent.formatters
import format_consensus` (and all the other format_xxx imports) keeps
working unchanged; callers don't need to know or care whether a given
formatter lives in this file or a submodule.
"""

from src.agent.formatters.consensus_formatter import format_consensus
from src.agent.formatters.valuation_formatter import format_valuation
from src.agent.formatters.snapshot_formatter import format_stock_snapshot


# ─────────────────────────────────────────────
# Raw data formatters (fetched data, not computed signals)
# ─────────────────────────────────────────────

# format_stock_snapshot() now lives in snapshot_formatter.py — see
# that file's docstring (same principle as format_consensus() /
# format_valuation(), 2026-07-27).



def format_sec_chunks(chunks: list, ticker: str) -> str:
    sec_context = f"\n{ticker} SEC Filing:\n"
    for i, chunk in enumerate(chunks):
        source = f"{ticker}_{chunk['chunk']['filing_type']}_{chunk['chunk']['section']}_{chunk['chunk']['filing_date']}"
        sec_context += f"[Source {i+1}: {source} | Score: {chunk['score']:.2f}]\n"
        sec_context += chunk["chunk"]["text"] + "\n"
    return sec_context


def format_conversation_context(messages: list, limit: int, max_chars: int = None) -> str:
    conversation_context = ""
    for msg in messages[-limit:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if max_chars:
            content = content[:max_chars]
        conversation_context += f"{role.upper()}: {content}\n"
    return conversation_context

# ─────────────────────────────────────────────
# Quant signal formatters (computed signal results, not raw data)
# ─────────────────────────────────────────────

# format_valuation() now lives in valuation_formatter.py — see that
# file's docstring (same principle as format_consensus() /
# consensus_formatter.py, 2026-07-27).


def format_momentum(mom: dict | None) -> str:
    """
    Two independent sub-signals, NOT combined into a single score
    (12-1 momentum answers "how much has it moved over the past year",
    52-week position answers "how close is it to its own high right
    now" — different questions, not to be averaged).
    """
    if not mom:
        return "  Momentum: Insufficient data (e.g. recent IPO with limited price history).\n"
    text = f"  Momentum - 12-1 Month Return: {mom['momentum_12_1_label']} ({mom['momentum_12_1_pct']}%, percentile={mom['momentum_12_1_percentile']})\n"
    text += f"  Momentum - 52-Week High Position: {mom['position_52w_label']} ({round(mom['position_52w']*100)}% of 52w range, percentile={mom['position_52w_percentile']})\n"
    text += f"  {mom['detail']}\n"
    if mom.get("stale_benchmark"):
        text += f"  ⚠️ Momentum benchmarks may be outdated (>90 days old).\n"
    return text


def format_risk(risk: dict | None) -> str:
    """
    Four independent sub-signals, NOT combined into a single score
    (Beta/Sharpe/VaR/Max Drawdown each answer a different risk question
    — averaging them would hide which dimension matters, e.g. a strong
    Sharpe Ratio can mask a catastrophic Max Drawdown).
    """
    if not risk:
        return "  Risk: Insufficient data.\n"
    text = ""
    if risk.get("beta"):
        text += f"  Risk - Beta: {risk['beta']['adjusted_beta']} (score={risk['beta']['beta_score']})\n"
    else:
        text += f"  Risk - Beta: Unavailable (market benchmark data missing).\n"
    text += f"  Risk - Sharpe Ratio: {risk['sharpe']['sharpe_ratio']} (score={risk['sharpe']['sharpe_score']})\n"
    text += f"  Risk - VaR (95%): {risk['var']['var_95']*100:.2f}% (score={risk['var']['var_score']})\n"
    text += f"  Risk - Max Drawdown: {risk['max_drawdown']['max_drawdown']*100:.2f}% (score={risk['max_drawdown']['drawdown_score']})\n"
    text += f"  {risk['detail']}\n"
    if risk.get("low_confidence"):
        text += f"  ⚠️ Risk signal based on less than 1 year of price history — lower confidence.\n"
    if risk.get("beta") is None:
        text += f"  ⚠️ Beta could not be computed — market benchmark data unavailable.\n"
    if risk.get("max_drawdown", {}).get("stress_tested") is False:
        text += f"  ⚠️ This stock has not experienced a significant decline in the observed window — risk may be understated.\n"
    return text


def format_quality(quality: dict | None) -> str:
    """
    Shows the F-Score summary plus all 9 signals grouped by outcome
    (met / not met / insufficient data), so the LLM and the user can see
    exactly WHICH signals drove the score, not just the aggregate count
    — needed to judge whether a low score reflects genuine deterioration
    or a deliberate short-term trade-off (per the detail note below).
    """
    if not quality:
        return "  Quality: Insufficient data.\n"
    text = f"  Quality: {quality['quality_label']} (F-Score={quality['f_score_raw']}/{quality['signals_evaluated']}, score={quality['quality_score']})\n"
    text += f"  {quality['detail']}\n"
    if quality.get("signals_evaluated", 9) < 9:
        text += f"  ⚠️ Only {quality['signals_evaluated']} of 9 F-Score signals could be evaluated due to missing financial data.\n"

    breakdown = quality.get("breakdown") or {}
    passed  = [(k, v) for k, v in breakdown.items() if v["score"] == 1]
    failed  = [(k, v) for k, v in breakdown.items() if v["score"] == 0]
    missing = [(k, v) for k, v in breakdown.items() if v["score"] is None]

    if passed:
        text += f"  Signals met ({len(passed)}):\n"
        for name, v in passed:
            text += f"    - {name}: {v['detail']}\n"
    if failed:
        text += f"  Signals not met ({len(failed)}):\n"
        for name, v in failed:
            text += f"    - {name}: {v['detail']}\n"
    if missing:
        text += f"  Signals with insufficient data ({len(missing)}):\n"
        for name, v in missing:
            text += f"    - {name}: {v['detail']}\n"
    return text


def format_news_sentiment(news_sentiment: dict | None) -> str:
    """
    Media tone on recent company-specific news, distinct from Consensus
    (professional analyst opinion) and a future Management Risk
    Sentiment Signal (10-K risk section tone).

    Includes the full per-article list (title, url, published, sentiment)
    -- this is the only place article-level news data reaches the LLM,
    since news_sentiment_signal's "articles" field already contains
    everything news_reader.py provides, plus the FinBERT sentiment label.
    """
    if not news_sentiment:
        return "  News Sentiment: Insufficient data.\n"
    text = f"  News Sentiment: {news_sentiment['sentiment_label']} (score={news_sentiment['sentiment_score']})\n"
    text += f"  {news_sentiment['detail']}\n"
    if news_sentiment.get("low_confidence"):
        text += f"  ⚠️ Low article count — reduced confidence in this signal.\n"

    articles = news_sentiment.get("articles") or []
    if articles:
        text += f"  Articles ({len(articles)}):\n"
        for a in articles:
            text += f"    - [{a['sentiment'].upper()}] (confidence={a['score']}) {a['title']}\n"
            text += f"      URL: {a['url']}\n"
            text += f"      Published: {a['published']}\n"
    return text


def format_financial_history(rows: list, ticker: str) -> str:
    """
    Formats multi-year financial_history rows (annual + quarterly) into
    a plain-text year-over-year table for the LLM. Only triggered when
    the user's question suggests they want historical trend data (see
    fetch_all_data.py's _question_wants_financial_history()) — not
    fetched unconditionally like the other 6 data sources, since most
    questions don't need multi-year figures.

    Shows total_revenue and net_income only (the two figures users most
    commonly ask about in a "historical trend" question) — not all 26
    stored metrics, to avoid overloading the prompt with numbers the
    question didn't ask about.
    """
    if not rows:
        return f"\n{ticker}: No historical financial data available.\n"

    annual_rows = [r for r in rows if r["period_type"] == "annual"]
    quarterly_rows = [r for r in rows if r["period_type"] == "quarterly"]

    text = f"\n{ticker} Historical Financials:\n"

    def _format_row(r):
        rev = r.get("total_revenue")
        ni = r.get("net_income")
        if rev is not None and ni is not None:
            return f"    {r['period_end']}: Revenue ${rev/1e9:.2f}B, Net Income ${ni/1e9:.2f}B\n"
        return f"    {r['period_end']}: Revenue/Net Income data incomplete\n"

    if annual_rows:
        text += "  Annual (fiscal year end):\n"
        for r in sorted(annual_rows, key=lambda x: x["period_end"]):
            text += _format_row(r)

    if quarterly_rows:
        text += "  Quarterly (most recent periods):\n"
        for r in sorted(quarterly_rows, key=lambda x: x["period_end"]):
            text += _format_row(r)

    return text


def format_quant_signals(signals: dict, ticker: str) -> str:
    """
    Formats all six quant signals for one ticker into a single block.
    Each signal is independently formatted (see format_valuation,
    format_momentum, format_risk, format_quality, format_news_sentiment,
    format_consensus in this package) — this function just concatenates
    them under one ticker heading.
    """
    text = f"\n{ticker} Quantitative Signals:\n"
    text += format_valuation(signals.get("valuation"))
    text += format_momentum(signals.get("momentum"))
    text += format_risk(signals.get("risk"))
    text += format_quality(signals.get("quality"))
    text += format_news_sentiment(signals.get("news_sentiment"))
    text += format_consensus(signals.get("consensus"))
    return text
