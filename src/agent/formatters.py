"""
src/agent/formatters.py

Shared formatters for converting structured data into LLM-readable strings.
Used by fetch_all_data.py and nodes.py to ensure consistent data presentation.
"""


# ─────────────────────────────────────────────
# Raw data formatters (fetched data, not computed signals)
# ─────────────────────────────────────────────

def format_stock_snapshot(data: dict, ticker: str) -> str:
    return f"""
{ticker} — {data.get('company_name')}
  Price: ${data.get('current_price')} | Market Cap: {data.get('market_cap')}
  P/E: {data.get('pe_ratio')} | Forward P/E: {data.get('forward_pe')}
  Revenue: {data.get('revenue')} | Profit Margin: {data.get('profit_margin')}
  EPS (TTM): {data.get('eps_trailing')} | EPS (Fwd): {data.get('eps_forward')}
  52w High: {data.get('52w_high')} | 52w Low: {data.get('52w_low')}
  Dividend Yield: {data.get('dividend_yield')}
  Analyst Target: ${data.get('target_mean')} (Low: ${data.get('target_low')} / High: ${data.get('target_high')}) | Recommendation: {data.get('recommendation')}
  Analyst Count: {data.get('analyst_count')} | Recommendation Mean: {data.get('recommendation_mean')} (1=Strong Buy, 3=Hold, 5=Strong Sell)
  Sector: {data.get('sector')} | Industry: {data.get('industry')}
"""



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

def format_valuation(val: dict | None) -> str:
    if not val:
        return "  Valuation: Insufficient data.\n"
    text = f"  Valuation: {val['valuation_label']} (score={val['valuation_score']}, method={val['method']})\n"
    text += f"  {val['detail']}\n"
    if val.get("peers_used"):
        text += f"  Compared against: {', '.join(val['peers_used'])}\n"
    if val.get("reference_only"):
        text += f"  ⚠️ Valuation reference only — company is loss-making, P/S used instead of P/E.\n"
    if val.get("stale_benchmark"):
        text += f"  ⚠️ Sector benchmarks may be outdated (>90 days old).\n"
    return text


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
    everything news_data.py provides, plus the FinBERT sentiment label.
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


def format_consensus(consensus: dict | None) -> str:
    """
    Three independent sub-signals, NOT combined into a single score
    (recommendation/upside/trend answer different questions: current
    standing, future price target, and recent directional change —
    averaging them would hide the full picture).
    """
    if not consensus:
        return "  Consensus: Insufficient data.\n"
    text = f"  Consensus - Recommendation: {consensus['recommendation_label']} (score={consensus['recommendation_score']})\n"
    text += f"  Consensus - Upside: {consensus['upside_label']} ({consensus['upside_pct']}% implied by analyst target price)\n"
    if consensus.get("trend_label") is not None:
        text += f"  Consensus - Trend: {consensus['trend_label']} (trend_score={consensus['trend_score']})\n"
    else:
        text += f"  Consensus - Trend: Insufficient rating history.\n"
    text += f"  {consensus['detail']}\n"
    if consensus.get("low_confidence"):
        text += f"  ⚠️ Low analyst sample size — reduced confidence.\n"
    if consensus.get("wide_dispersion"):
        text += f"  ⚠️ Wide analyst target price dispersion — significant disagreement.\n"
    return text


def format_quant_signals(signals: dict, ticker: str) -> str:
    """
    Formats all six quant signals for one ticker into a single block.
    Each signal is independently formatted (see format_valuation,
    format_momentum, format_risk, format_quality, format_news_sentiment,
    format_consensus above) — this function just concatenates them under
    one ticker heading.
    """
    text = f"\n{ticker} Quantitative Signals:\n"
    text += format_valuation(signals.get("valuation"))
    text += format_momentum(signals.get("momentum"))
    text += format_risk(signals.get("risk"))
    text += format_quality(signals.get("quality"))
    text += format_news_sentiment(signals.get("news_sentiment"))
    text += format_consensus(signals.get("consensus"))
    return text
