"""
src/agent/formatters/news_sentiment_formatter.py

Formats news_sentiment_signal() results for the LLM — see
src/quant/news_sentiment_signal.py for the data this consumes.

Split out from the former single-file formatters.py on 2026-07-27,
alongside consensus_formatter.py, valuation_formatter.py,
snapshot_formatter.py, and momentum_formatter.py.

Adds an explicit positive/negative/neutral count summary line (not
present before 2026-07-27) — the full per-article list was already
shown, but without a stated count, the LLM had to manually tally
sentiment labels across potentially dozens of articles (e.g. 89 for
SPCX) to report totals, an unnecessary and error-prone step when the
signal already computed these counts (same principle as exposing raw
inputs in consensus_formatter.py / valuation_formatter.py: don't make
the LLM re-derive something the signal already knows).
"""


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
    text += (
        f"  Article breakdown: {news_sentiment['total_articles']} total "
        f"({news_sentiment['positive_count']} positive, "
        f"{news_sentiment['negative_count']} negative, "
        f"{news_sentiment['neutral_count']} neutral).\n"
    )
    text += f"  {news_sentiment['detail']}\n"
    if news_sentiment.get("low_confidence"):
        text += f"  Note: Low article count — reduced confidence in this signal.\n"

    articles = news_sentiment.get("articles") or []
    if articles:
        text += f"  Articles ({len(articles)}):\n"
        for a in articles:
            text += f"    - [{a['sentiment'].upper()}] (confidence={a['score']}) {a['title']}\n"
            text += f"      URL: {a['url']}\n"
            text += f"      Published: {a['published']}\n"
    return text
