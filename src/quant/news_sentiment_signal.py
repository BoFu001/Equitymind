"""
src/quant/news_sentiment_signal.py

News Sentiment Signal — Layer 2 Quantitative Intelligence.

Pure compute-layer module: scores already-fetched, already-filtered
news articles with FinBERT and aggregates them into a single sentiment
signal. Does no data fetching and no relevance filtering — both of
those are handled upstream by src/readers/news_reader.py, consistent with
the project's data-layer / compute-layer separation.

This measures MEDIA sentiment specifically — how outlets are currently
writing about the company's own news and events. This is distinct from
Consensus Signal (professional analyst opinion) and is intended to
eventually sit alongside a separate Management Risk Sentiment Signal
(how the company itself talks about its own risks in SEC filings) —
three independent viewpoints, not to be combined into one score.
"""

from transformers import pipeline

# Load FinBERT sentiment model once at module level
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    truncation=True,
)

# Below this article count, the sentiment signal is flagged as low
# confidence — sample size is too small to be reliable. Chosen based on
# observed real data: large-caps typically had 60-75 relevant articles
# in 30 days, while small/mid-caps had only 5-10 — a full order of
# magnitude fewer.
MIN_ARTICLE_COUNT = 10


def _score_article(title: str, summary: str) -> tuple[str, float]:
    """
    Scores a single article's title + summary with FinBERT.
    Returns (label, score) — label is "positive"/"negative"/"neutral",
    score is the model's confidence in that label (0.0 to 1.0).
    """
    # No manual character-count truncation here — the pipeline's
    # truncation=True (see module-level `finbert = pipeline(...)`)
    # truncates by actual TOKEN count (BERT's real 512-token limit),
    # not by an approximated character count. A fixed [:512] character
    # slice was cutting text far shorter than the model could actually
    # handle (roughly 350-400 English words fit in 512 tokens, vs. only
    # ~80-100 words in 512 characters) — 2026-07-27, discovered while
    # investigating FinBERT misclassifications on longer article
    # summaries.
    text = f"{title}. {summary}"
    try:
        result = finbert(text)[0]
        return result["label"], round(result["score"], 4)
    except Exception:
        return "neutral", 0.0


def _aggregate_sentiment(scored_articles: list) -> dict:
    """
    Aggregates a list of FinBERT-scored articles into a single net
    sentiment score and counts.

    net_score is the average of each article's signed confidence score
    (positive articles contribute +score, negative articles contribute
    -score, neutral articles contribute 0), clipped to [-1, 1] for
    consistency with every other signal engine in this system.
    """
    total = len(scored_articles)

    positive = [a for a in scored_articles if a["sentiment"] == "positive"]
    negative = [a for a in scored_articles if a["sentiment"] == "negative"]
    neutral  = [a for a in scored_articles if a["sentiment"] == "neutral"]

    net_score = sum(
        a["score"] if a["sentiment"] == "positive" else
        -a["score"] if a["sentiment"] == "negative" else
        0
        for a in scored_articles
    ) / total

    net_score = round(max(-1.0, min(1.0, net_score)), 4)

    return {
        "net_score":      net_score,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "neutral_count":  len(neutral),
        "total_articles": total,
    }


def _sentiment_label(net_score: float) -> str:
    """
    Maps the net sentiment score to a human-readable label. Uses a
    neutral buffer zone (-0.15 to +0.15) so that scores very close to
    zero aren't labeled as a strong positive or negative — consistent
    with the threshold style used in consensus_signal.py.
    """
    if net_score > 0.15:
        return "positive"
    if net_score < -0.15:
        return "negative"
    return "neutral"


def news_sentiment_signal(ticker: str, articles: list) -> dict | None:
    """
    Scores and aggregates already-filtered news articles into a single
    News Sentiment Signal.

    Args:
        ticker: stock ticker symbol, e.g. "TSLA" — used only for logging,
                not for any filtering (articles are already filtered by
                the time they reach this function — see news_reader.py)
        articles: list of dicts from fetch_company_news(), each with at
                  least "title" and "summary" keys — already filtered
                  to company-specific articles, not yet scored

    Returns:
        dict with keys:
            - sentiment_score:  float (-1.0 to +1.0)
            - sentiment_label:  str — "positive" / "neutral" / "negative"
            - positive_count:   int
            - negative_count:   int
            - neutral_count:    int
            - total_articles:   int
            - low_confidence:   bool — True if total_articles < MIN_ARTICLE_COUNT
            - detail:           str — plain English explanation
            - articles:         list — each input article, with "sentiment"
                                 and "score" added (title, url, published,
                                 summary, sentiment, score) — for callers
                                 that need to display individual articles
                                 (e.g. a source list with links)
        or None if articles is empty
    """
    if not articles:
        return None

    scored_articles = []
    for article in articles:
        label, score = _score_article(article.get("title", ""), article.get("summary", ""))
        scored_articles.append({**article, "sentiment": label, "score": score})

    stats = _aggregate_sentiment(scored_articles)
    label = _sentiment_label(stats["net_score"])
    low_confidence = stats["total_articles"] < MIN_ARTICLE_COUNT

    detail = (
        f"Of {stats['total_articles']} company-specific news articles in "
        f"the past 30 days, {stats['positive_count']} were positive, "
        f"{stats['negative_count']} negative, and {stats['neutral_count']} "
        f"neutral (net sentiment score: {stats['net_score']})."
    )
    if low_confidence:
        detail += (
            f" \u26a0\ufe0f Only {stats['total_articles']} company-specific "
            f"articles were found — small sample size, lower confidence "
            f"in this signal."
        )

    print(f"  [news_sentiment_signal] {ticker}: {label} (score={stats['net_score']}, n={stats['total_articles']})")

    return {
        "sentiment_score": stats["net_score"],
        "sentiment_label": label,
        "positive_count":  stats["positive_count"],
        "negative_count":  stats["negative_count"],
        "neutral_count":   stats["neutral_count"],
        "total_articles":  stats["total_articles"],
        "low_confidence":  low_confidence,
        "detail":          detail,
        "articles":        scored_articles,
    }
